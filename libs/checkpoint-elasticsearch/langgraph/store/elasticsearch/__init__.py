"""Synchronous Elasticsearch store for LangGraph."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
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
    _build_filter_clauses,
    _doc_id,
    _ensure_index_config,
    _expires_iso,
    _group_ops,
    _hit_to_item,
    _hit_to_search_item,
    _namespace_to_text,
    _not_expired_filter,
    _now_iso,
    _text_to_namespace,
)

logger = logging.getLogger(__name__)


class ElasticsearchStore(BaseStore, BaseElasticsearchStore):
    """Elasticsearch-backed store for LangGraph with optional vector search.

    Example::

        from elasticsearch import Elasticsearch
        from langgraph.store.elasticsearch import ElasticsearchStore

        es = Elasticsearch("http://localhost:9200")
        store = ElasticsearchStore(es)
        store.setup()

        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")

    With vector search::

        from langchain.embeddings import init_embeddings

        store = ElasticsearchStore(
            es,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"],
            }
        )
        store.setup()
        store.put(("docs",), "doc1", {"text": "Python tutorial"})
        results = store.search(("docs",), query="programming guides")

    Note:
        Call ``setup()`` once before first use.

    Note:
        To enable automatic expiry of items, provide a ``ttl`` config and call
        ``start_ttl_sweeper()``.  Call ``stop_ttl_sweeper()`` on shutdown.
    """

    __slots__ = (
        "_conn",
        "lock",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
    )

    supports_ttl: bool = True

    def __init__(
        self,
        conn: Elasticsearch,
        *,
        index: ESIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self._conn = conn
        self.lock = threading.Lock()
        self.ttl_config = ttl
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

        if index is not None:
            self.embeddings, self.index_config = _ensure_index_config(index)
        else:
            self.embeddings = None
            self.index_config = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        hosts: str | list[str],
        *,
        index: ESIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        **es_kwargs: Any,
    ) -> Iterator[ElasticsearchStore]:
        """Create a store from connection parameters.

        Args:
            hosts: Elasticsearch hosts, e.g. ``"http://localhost:9200"``.
            index: Optional vector index configuration.
            ttl: Optional TTL configuration.
            **es_kwargs: Extra kwargs forwarded to ``Elasticsearch()``.
        """
        conn = Elasticsearch(hosts, **es_kwargs)
        try:
            yield cls(conn, index=index, ttl=ttl)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create the store index if it does not already exist."""
        body = self._build_index_body()
        if not self._conn.indices.exists(index=STORE_INDEX):
            self._conn.indices.create(index=STORE_INDEX, body=body)
        elif self.index_config:
            # Ensure dense_vector field is present (idempotent put mapping)
            dims = self.index_config.get("dims")
            similarity = cast(dict, self.index_config).get("similarity", "cosine")
            if dims:
                try:
                    self._conn.indices.put_mapping(
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
                    pass  # field already exists with compatible mapping

    # ------------------------------------------------------------------
    # BaseStore.batch — the single required method
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped:
            self._batch_get(cast(list, grouped[GetOp]), results)

        if SearchOp in grouped:
            self._batch_search(cast(list, grouped[SearchOp]), results)

        if ListNamespacesOp in grouped:
            self._batch_list_namespaces(cast(list, grouped[ListNamespacesOp]), results)

        if PutOp in grouped:
            self._batch_put(cast(list, grouped[PutOp]))

        return results

    # ------------------------------------------------------------------
    # abatch — run batch in a thread executor
    # ------------------------------------------------------------------

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch, list(ops))

    # ------------------------------------------------------------------
    # Internal batch handlers
    # ------------------------------------------------------------------

    def _batch_get(
        self,
        get_ops: list[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        # Group by namespace so we can batch per-namespace mget calls.
        ns_groups: dict[tuple[str, ...], list[tuple[int, str]]] = {}
        refresh_ttls: dict[tuple[str, ...], list[bool]] = {}
        for idx, op in get_ops:
            ns_groups.setdefault(op.namespace, []).append((idx, op.key))
            refresh_ttls.setdefault(op.namespace, []).append(op.refresh_ttl)

        for namespace, items in ns_groups.items():
            prefix = _namespace_to_text(namespace)
            doc_ids = [_doc_id(prefix, key) for _, key in items]
            resp = self._conn.mget(index=STORE_INDEX, body={"ids": doc_ids})

            # Build refresh list
            to_refresh: list[str] = []
            for (idx, key), doc, should_refresh in zip(
                items, resp["docs"], refresh_ttls[namespace], strict=False
            ):
                if doc.get("found"):
                    item = _hit_to_item(doc)
                    results[idx] = item
                    if should_refresh and doc["_source"].get("ttl_minutes") is not None:
                        to_refresh.append(doc["_id"])
                else:
                    results[idx] = None

            if to_refresh:
                self._conn.update_by_query(
                    index=STORE_INDEX,
                    body=self._build_ttl_refresh_script(to_refresh),
                    refresh=False,
                )

    def _batch_put(self, put_ops: list[tuple[int, PutOp]]) -> None:
        actions, embedding_requests = self._build_put_actions(put_ops)

        if embedding_requests and self.embeddings is not None:
            texts = [r[3] for r in embedding_requests]
            vectors = self.embeddings.embed_documents(texts)
            # Patch embeddings into the corresponding actions
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
            bulk(self._conn, actions, raise_on_error=True)

    def _batch_search(
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
            vectors = self.embeddings.embed_documents(texts)
            for (i, _), vec in zip(embedding_requests, vectors, strict=False):
                embeddings_map[i] = vec

        for i, (idx, op) in enumerate(search_ops):
            embedding = embeddings_map.get(i)
            body = self._build_search_query(op, embedding=embedding)
            resp = self._conn.search(index=STORE_INDEX, body=body)
            hits = resp["hits"]["hits"]
            items: list[SearchItem] = [_hit_to_search_item(h) for h in hits]
            results[idx] = items

            # Refresh TTL for returned items if requested
            if op.refresh_ttl and items:
                ids = [h["_id"] for h in hits]
                self._conn.update_by_query(
                    index=STORE_INDEX,
                    body=self._build_ttl_refresh_script(ids),
                    refresh=False,
                )

    def _batch_list_namespaces(
        self,
        list_ops: list[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        for idx, op in list_ops:
            body = self._build_list_ns_query(op)
            resp = self._conn.search(index=STORE_INDEX, body=body)
            results[idx] = self._process_list_ns_response(resp, op)

    # ------------------------------------------------------------------
    # TTL sweeper
    # ------------------------------------------------------------------

    def sweep_ttl(self) -> int:
        """Delete items whose TTL has expired.  Returns the number deleted."""
        resp = self._conn.delete_by_query(
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

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Start a background thread that periodically sweeps expired items."""
        if not self.ttl_config:
            f: concurrent.futures.Future[None] = concurrent.futures.Future()
            f.set_result(None)
            return f

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            f = concurrent.futures.Future()
            f.add_done_callback(
                lambda fut: self._ttl_stop_event.set() if fut.cancelled() else None
            )
            return f

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes
            or self.ttl_config.get("sweep_interval_minutes")
            or 5
        )
        logger.info("Starting ES store TTL sweeper, interval=%s min", interval)
        f = concurrent.futures.Future()

        def _loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        n = self.sweep_ttl()
                        if n:
                            logger.info("ES store swept %d expired item(s)", n)
                    except Exception as exc:
                        logger.exception("ES store TTL sweep failed", exc_info=exc)
                f.set_result(None)
            except Exception as exc:
                f.set_exception(exc)

        t = threading.Thread(target=_loop, daemon=True, name="es-store-ttl-sweeper")
        self._ttl_sweeper_thread = t
        t.start()
        f.add_done_callback(
            lambda fut: self._ttl_stop_event.set() if fut.cancelled() else None
        )
        return f

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread.  Returns True if cleanly stopped."""
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        stopped = not self._ttl_sweeper_thread.is_alive()
        if stopped:
            self._ttl_sweeper_thread = None
        return stopped

    def __del__(self) -> None:
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)
