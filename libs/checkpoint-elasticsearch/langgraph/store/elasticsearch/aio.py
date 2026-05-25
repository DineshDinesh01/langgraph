"""Asynchronous Elasticsearch store for LangGraph."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, cast

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

from langgraph.store.elasticsearch.base import (
    STORE_INDEX,
    BaseElasticsearchStore,
    ESIndexConfig,
    _doc_id,
    _ensure_index_config,
    _group_ops,
    _hit_to_item,
    _hit_to_search_item,
    _namespace_to_text,
)

logger = logging.getLogger(__name__)


class AsyncElasticsearchStore(BaseStore, BaseElasticsearchStore):
    """Async Elasticsearch-backed store for LangGraph with optional vector search.

    Example::

        from elasticsearch import AsyncElasticsearch
        from langgraph.store.elasticsearch.aio import AsyncElasticsearchStore

        async def main():
            es = AsyncElasticsearch("http://localhost:9200")
            store = AsyncElasticsearchStore(es)
            await store.setup()

            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")

    Note:
        Call ``await setup()`` once before first use.
    """

    __slots__ = (
        "_conn",
        "lock",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_task",
        "_ttl_stop_event",
    )

    supports_ttl: bool = True

    def __init__(
        self,
        conn: AsyncElasticsearch,
        *,
        index: ESIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self._conn = conn
        self.lock = asyncio.Lock()
        self.ttl_config = ttl
        self._ttl_task: asyncio.Task | None = None
        self._ttl_stop_event: asyncio.Event | None = None

        if index is not None:
            self.embeddings, self.index_config = _ensure_index_config(index)
        else:
            self.embeddings = None
            self.index_config = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        hosts: str | list[str],
        *,
        index: ESIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        **es_kwargs: Any,
    ):
        """Create a store from connection parameters."""
        conn = AsyncElasticsearch(hosts, **es_kwargs)
        try:
            yield cls(conn, index=index, ttl=ttl)
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create the store index if it does not already exist."""
        body = self._build_index_body()
        if not await self._conn.indices.exists(index=STORE_INDEX):
            await self._conn.indices.create(index=STORE_INDEX, body=body)
        elif self.index_config:
            dims = self.index_config.get("dims")
            similarity = cast(dict, self.index_config).get("similarity", "cosine")
            if dims:
                try:
                    await self._conn.indices.put_mapping(
                        index=STORE_INDEX,
                        body={
                            "properties": {
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": dims,
                                    "index": True,
                                    "similarity": similarity,
                                }
                            }
                        },
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # BaseStore.batch / abatch
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Sync shim — runs abatch in a thread-safe executor."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot call sync batch() from a running event loop. "
                "Use await abatch() instead."
            )
        return asyncio.run(self.abatch(list(ops)))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped:
            await self._batch_get(cast(list, grouped[GetOp]), results)

        if SearchOp in grouped:
            await self._batch_search(cast(list, grouped[SearchOp]), results)

        if ListNamespacesOp in grouped:
            await self._batch_list_namespaces(
                cast(list, grouped[ListNamespacesOp]), results
            )

        if PutOp in grouped:
            await self._batch_put(cast(list, grouped[PutOp]))

        return results

    # ------------------------------------------------------------------
    # Internal batch handlers (async)
    # ------------------------------------------------------------------

    async def _batch_get(
        self,
        get_ops: list[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        ns_groups: dict[tuple[str, ...], list[tuple[int, str]]] = {}
        refresh_ttls: dict[tuple[str, ...], list[bool]] = {}
        for idx, op in get_ops:
            ns_groups.setdefault(op.namespace, []).append((idx, op.key))
            refresh_ttls.setdefault(op.namespace, []).append(op.refresh_ttl)

        for namespace, items in ns_groups.items():
            prefix = _namespace_to_text(namespace)
            doc_ids = [_doc_id(prefix, key) for _, key in items]
            resp = await self._conn.mget(index=STORE_INDEX, body={"ids": doc_ids})

            to_refresh: list[str] = []
            for (idx, key), doc, should_refresh in zip(
                items, resp["docs"], refresh_ttls[namespace], strict=False
            ):
                if doc.get("found"):
                    results[idx] = _hit_to_item(doc)
                    if should_refresh and doc["_source"].get("ttl_minutes") is not None:
                        to_refresh.append(doc["_id"])
                else:
                    results[idx] = None

            if to_refresh:
                await self._conn.update_by_query(
                    index=STORE_INDEX,
                    body=self._build_ttl_refresh_script(to_refresh),
                    refresh=False,
                )

    async def _batch_put(self, put_ops: list[tuple[int, PutOp]]) -> None:
        actions, embedding_requests = self._build_put_actions(put_ops)

        if embedding_requests and self.embeddings is not None:
            texts = [r[3] for r in embedding_requests]
            # Try async embed first, fall back to sync
            if hasattr(self.embeddings, "aembed_documents"):
                vectors = await self.embeddings.aembed_documents(texts)
            else:
                loop = asyncio.get_running_loop()
                vectors = await loop.run_in_executor(
                    None, self.embeddings.embed_documents, texts
                )
            doc_id_to_vector = {
                r[0]: v for r, v in zip(embedding_requests, vectors, strict=False)
            }
            for action in actions:
                if action.get("_op_type") == "update":
                    aid = action["_id"]
                    if aid in doc_id_to_vector:
                        action["script"]["params"]["doc"]["embedding"] = (
                            doc_id_to_vector[aid]
                        )
                        action["upsert"]["embedding"] = doc_id_to_vector[aid]

        if actions:
            await async_bulk(self._conn, actions, raise_on_error=True)

    async def _batch_search(
        self,
        search_ops: list[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        embedding_requests: list[tuple[int, str]] = []
        for i, (_, op) in enumerate(search_ops):
            if op.query and self.index_config and self.embeddings is not None:
                embedding_requests.append((i, op.query))

        embeddings_map: dict[int, list[float]] = {}
        if embedding_requests:
            texts = [text for _, text in embedding_requests]
            if hasattr(self.embeddings, "aembed_documents"):
                vectors = await self.embeddings.aembed_documents(texts)
            else:
                loop = asyncio.get_running_loop()
                vectors = await loop.run_in_executor(
                    None, self.embeddings.embed_documents, texts
                )
            for (i, _), vec in zip(embedding_requests, vectors, strict=False):
                embeddings_map[i] = vec

        for i, (idx, op) in enumerate(search_ops):
            embedding = embeddings_map.get(i)
            body = self._build_search_query(op, embedding=embedding)
            resp = await self._conn.search(index=STORE_INDEX, body=body)
            hits = resp["hits"]["hits"]
            items: list[SearchItem] = [_hit_to_search_item(h) for h in hits]
            results[idx] = items

            if op.refresh_ttl and items:
                ids = [h["_id"] for h in hits]
                await self._conn.update_by_query(
                    index=STORE_INDEX,
                    body=self._build_ttl_refresh_script(ids),
                    refresh=False,
                )

    async def _batch_list_namespaces(
        self,
        list_ops: list[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        for idx, op in list_ops:
            body = self._build_list_ns_query(op)
            resp = await self._conn.search(index=STORE_INDEX, body=body)
            results[idx] = self._process_list_ns_response(resp, op)

    # ------------------------------------------------------------------
    # TTL sweeper (async)
    # ------------------------------------------------------------------

    async def sweep_ttl(self) -> int:
        """Delete items whose TTL has expired.  Returns the number deleted."""
        resp = await self._conn.delete_by_query(
            index=STORE_INDEX,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"exists": {"field": "expires_at"}},
                            {"range": {"expires_at": {"lt": "now"}}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return int(resp.get("deleted", 0))

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> None:
        """Start a background async task that periodically sweeps expired items."""
        if not self.ttl_config:
            return
        if self._ttl_task and not self._ttl_task.done():
            return

        loop = asyncio.get_running_loop()
        self._ttl_stop_event = asyncio.Event()
        interval = float(
            sweep_interval_minutes
            or self.ttl_config.get("sweep_interval_minutes")
            or 5
        )
        stop_event = self._ttl_stop_event

        async def _loop() -> None:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=interval * 60
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                try:
                    n = await self.sweep_ttl()
                    if n:
                        logger.info("ES async store swept %d expired item(s)", n)
                except Exception as exc:
                    logger.exception(
                        "ES async store TTL sweep failed", exc_info=exc
                    )

        self._ttl_task = loop.create_task(_loop())

    async def stop_ttl_sweeper(self) -> None:
        """Stop the TTL sweeper task."""
        if self._ttl_stop_event:
            self._ttl_stop_event.set()
        if self._ttl_task:
            try:
                await asyncio.wait_for(self._ttl_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._ttl_task.cancel()
            self._ttl_task = None
