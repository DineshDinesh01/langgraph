"""Asynchronous Elasticsearch checkpoint saver for LangGraph."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS, _DeltaSnapshot

from langgraph.checkpoint.elasticsearch._ainternal import Conn
from langgraph.checkpoint.elasticsearch.base import (
    _DELTA_PAGE_SIZE,
    CHECKPOINT_BLOBS_INDEX,
    CHECKPOINT_INDEX,
    CHECKPOINT_INDEX_MAPPING,
    CHECKPOINT_BLOBS_INDEX_MAPPING,
    CHECKPOINT_WRITES_INDEX,
    CHECKPOINT_WRITES_INDEX_MAPPING,
    BaseElasticsearchSaver,
    _make_blob_doc_id,
    _make_checkpoint_doc_id,
)


class AsyncElasticsearchSaver(BaseElasticsearchSaver):
    """Async checkpointer that stores LangGraph checkpoints in Elasticsearch.

    Usage::

        from langgraph.checkpoint.elasticsearch.aio import AsyncElasticsearchSaver

        async with AsyncElasticsearchSaver.from_conn_string("http://localhost:9200") as saver:
            await saver.setup()
            graph = compiled_graph.compile(checkpointer=saver)
            result = await graph.ainvoke(inputs, {"configurable": {"thread_id": "t1"}})

    Synchronous methods (get_tuple, list, put, put_writes, delete_thread) are also
    provided for compatibility — they delegate to async via run_coroutine_threadsafe
    and must be called from a thread that is NOT the event-loop thread.
    """

    def __init__(
        self,
        conn: Conn,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        hosts: str | list[str],
        *,
        serde: SerializerProtocol | None = None,
        **es_kwargs: Any,
    ) -> AsyncIterator[AsyncElasticsearchSaver]:
        """Create an AsyncElasticsearchSaver from a host string or list.

        Args:
            hosts: ES host URL(s), e.g. "http://localhost:9200".
            serde: Optional custom serializer.
            **es_kwargs: Extra kwargs forwarded to AsyncElasticsearch.
        """
        client = AsyncElasticsearch(hosts, **es_kwargs)
        try:
            yield cls(client, serde=serde)
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create the three ES indices if they do not already exist."""
        for index, mapping in [
            (CHECKPOINT_INDEX, CHECKPOINT_INDEX_MAPPING),
            (CHECKPOINT_BLOBS_INDEX, CHECKPOINT_BLOBS_INDEX_MAPPING),
            (CHECKPOINT_WRITES_INDEX, CHECKPOINT_WRITES_INDEX_MAPPING),
        ]:
            if not await self.conn.indices.exists(index=index):
                await self.conn.indices.create(index=index, body=mapping)

    # ------------------------------------------------------------------
    # aget_tuple
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple asynchronously."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with self.lock:
            if checkpoint_id:
                doc_id = _make_checkpoint_doc_id(thread_id, checkpoint_ns, checkpoint_id)
                resp = await self.conn.get(
                    index=CHECKPOINT_INDEX, id=doc_id, ignore=[404]
                )
                if not resp.get("found"):
                    return None
                checkpoint_doc = resp
            else:
                resp = await self.conn.search(
                    index=CHECKPOINT_INDEX,
                    body={
                        "query": {
                            "bool": {
                                "filter": [
                                    {"term": {"thread_id": thread_id}},
                                    {"term": {"checkpoint_ns": checkpoint_ns}},
                                ]
                            }
                        },
                        "sort": [{"checkpoint_id": "desc"}],
                        "size": 1,
                    },
                )
                hits = resp["hits"]["hits"]
                if not hits:
                    return None
                checkpoint_doc = hits[0]

            src = checkpoint_doc["_source"]
            found_checkpoint_id: str = src["checkpoint_id"]
            parent_checkpoint_id: str | None = src.get("parent_checkpoint_id")

            blob_ids = self._blob_ids_for_checkpoint(thread_id, checkpoint_ns, src)
            if blob_ids:
                mget_resp = await self.conn.mget(
                    index=CHECKPOINT_BLOBS_INDEX, body={"ids": blob_ids}
                )
                blob_docs = mget_resp["docs"]
            else:
                blob_docs = []

            writes_resp = await self.conn.search(
                index=CHECKPOINT_WRITES_INDEX,
                body={
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"thread_id": thread_id}},
                                {"term": {"checkpoint_ns": checkpoint_ns}},
                                {"term": {"checkpoint_id": found_checkpoint_id}},
                            ]
                        }
                    },
                    "size": 10000,
                },
            )
            write_hits = writes_resp["hits"]["hits"]

            checkpoint_data = src.get("checkpoint", {})
            if checkpoint_data.get("v", 4) < 4 and parent_checkpoint_id:
                sends_resp = await self.conn.search(
                    index=CHECKPOINT_WRITES_INDEX,
                    body={
                        "query": {
                            "bool": {
                                "filter": [
                                    {"term": {"thread_id": thread_id}},
                                    {"term": {"checkpoint_id": parent_checkpoint_id}},
                                    {"term": {"channel": TASKS}},
                                ]
                            }
                        },
                        "sort": [
                            {"task_path": "asc"},
                            {"task_id": "asc"},
                            {"idx": "asc"},
                        ],
                        "size": 10000,
                    },
                )
                if sends_resp["hits"]["hits"]:
                    channel_values = checkpoint_data.setdefault("channel_values", {})
                    self._migrate_pending_sends(
                        sends_resp["hits"]["hits"],
                        checkpoint_data,
                        channel_values,
                    )

        return self._load_checkpoint_tuple(checkpoint_doc, blob_docs, write_hits)

    # ------------------------------------------------------------------
    # alist
    # ------------------------------------------------------------------

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints matching the given criteria, newest first (async)."""
        clauses = self._search_filters(config, filter, before)
        query = {"bool": {"filter": clauses}} if clauses else {"match_all": {}}
        body: dict[str, Any] = {
            "query": query,
            "sort": [{"checkpoint_id": "desc"}],
            "size": limit if limit is not None else 100,
        }

        async with self.lock:
            resp = await self.conn.search(index=CHECKPOINT_INDEX, body=body)
            checkpoint_docs = resp["hits"]["hits"]
            if not checkpoint_docs:
                return

            all_blob_ids: list[str] = []
            for doc in checkpoint_docs:
                src = doc["_source"]
                all_blob_ids.extend(
                    self._blob_ids_for_checkpoint(
                        src["thread_id"], src["checkpoint_ns"], src
                    )
                )

            blob_map: dict[str, dict[str, Any]] = {}
            if all_blob_ids:
                mget_resp = await self.conn.mget(
                    index=CHECKPOINT_BLOBS_INDEX, body={"ids": all_blob_ids}
                )
                for blob_doc in mget_resp["docs"]:
                    if blob_doc.get("found"):
                        blob_map[blob_doc["_id"]] = blob_doc

            checkpoint_ids = [d["_source"]["checkpoint_id"] for d in checkpoint_docs]
            writes_resp = await self.conn.search(
                index=CHECKPOINT_WRITES_INDEX,
                body={
                    "query": {
                        "bool": {
                            "filter": [{"terms": {"checkpoint_id": checkpoint_ids}}]
                        }
                    },
                    "size": 10000,
                },
            )
            writes_by_cid: dict[str, list[dict[str, Any]]] = {}
            for hit in writes_resp["hits"]["hits"]:
                cid = hit["_source"]["checkpoint_id"]
                writes_by_cid.setdefault(cid, []).append(hit)

        for doc in checkpoint_docs:
            src = doc["_source"]
            thread_id = src["thread_id"]
            checkpoint_ns = src["checkpoint_ns"]
            found_cid = src["checkpoint_id"]
            blob_ids = self._blob_ids_for_checkpoint(thread_id, checkpoint_ns, src)
            blob_docs = [blob_map[bid] for bid in blob_ids if bid in blob_map]
            write_hits = writes_by_cid.get(found_cid, [])
            yield self._load_checkpoint_tuple(doc, blob_docs, write_hits)

    # ------------------------------------------------------------------
    # aput
    # ------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Elasticsearch asynchronously."""
        configurable = config["configurable"].copy()
        thread_id: str = configurable.pop("thread_id")
        checkpoint_ns: str = configurable.pop("checkpoint_ns")
        parent_checkpoint_id: str | None = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        blob_values: dict[str, Any] = {}
        for k, v in checkpoint["channel_values"].items():
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy["channel_values"].pop(k)
                copy["channel_values"][k] = True
            elif v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}

        async with self.lock:
            if blob_versions:
                actions = self._dump_blobs(
                    thread_id, checkpoint_ns, blob_values, blob_versions
                )
                await async_bulk(self.conn, actions, raise_on_error=False)

            doc_id = _make_checkpoint_doc_id(
                thread_id, checkpoint_ns, checkpoint["id"]
            )
            await self.conn.index(
                index=CHECKPOINT_INDEX,
                id=doc_id,
                body={
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "type": checkpoint.get("type"),
                    "checkpoint": copy,
                    "metadata": get_serializable_checkpoint_metadata(config, metadata),
                },
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ------------------------------------------------------------------
    # aput_writes
    # ------------------------------------------------------------------

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate task writes asynchronously."""
        actions = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if not actions:
            return
        async with self.lock:
            await async_bulk(self.conn, actions, raise_on_error=False)

    # ------------------------------------------------------------------
    # adelete_thread
    # ------------------------------------------------------------------

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, blobs, and writes for a thread (async)."""
        query = {"query": {"term": {"thread_id": thread_id}}}
        async with self.lock:
            for index in [
                CHECKPOINT_INDEX,
                CHECKPOINT_BLOBS_INDEX,
                CHECKPOINT_WRITES_INDEX,
            ]:
                await self.conn.delete_by_query(
                    index=index, body=query, refresh=True
                )

    # ------------------------------------------------------------------
    # aget_delta_channel_history
    # ------------------------------------------------------------------

    async def aget_delta_channel_history(
        self,
        *,
        config: RunnableConfig,
        channels: Sequence[str],
    ) -> Mapping[str, DeltaChannelHistory]:
        """Retrieve channel write history asynchronously (two-stage walk)."""
        if not channels:
            return {}

        channels = list(channels)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with self.lock:
            if checkpoint_id is None:
                target = await self.aget_tuple(config)
                if target is None:
                    return {ch: {"writes": []} for ch in channels}
                checkpoint_id = target.config["configurable"]["checkpoint_id"]

            parent_of: dict[str, str | None] = {}
            ver_by_i_by_cid: list[dict[str, str | None]] = [{} for _ in channels]
            hs_by_i_by_cid: list[dict[str, bool]] = [{} for _ in channels]
            chain_by_ch: dict[str, list[str]] = {ch: [] for ch in channels}
            seed_ver_by_ch: dict[str, str | None] = {ch: None for ch in channels}
            walk_cursor_by_ch: dict[str, str | None] = {}
            seeded: set[str] = set()
            cursor: str | None = None

            while True:
                stage1_body = self._build_delta_stage1_query(
                    thread_id, checkpoint_ns, cursor
                )
                resp = await self.conn.search(
                    index=CHECKPOINT_INDEX, body=stage1_body
                )
                page_hits = resp["hits"]["hits"]
                if not page_hits:
                    break

                rows = [self._flatten_stage1_hit(h) for h in page_hits]
                oldest = self._ingest_stage1_page(
                    rows, channels, parent_of, ver_by_i_by_cid, hs_by_i_by_cid
                )
                self._try_advance_walks(
                    checkpoint_id,
                    channels,
                    parent_of,
                    ver_by_i_by_cid,
                    hs_by_i_by_cid,
                    chain_by_ch,
                    seed_ver_by_ch,
                    walk_cursor_by_ch,
                    seeded,
                )
                if len(seeded) == len(channels) or len(page_hits) < _DELTA_PAGE_SIZE:
                    break
                cursor = oldest

            channels_with_chain = [ch for ch in channels if chain_by_ch[ch]]
            channels_with_seed = [
                ch for ch in channels if seed_ver_by_ch.get(ch) is not None
            ]

            write_hits: list[dict[str, Any]] = []
            blob_docs: list[dict[str, Any]] = []

            # Parallelise writes search + seed blobs mget.
            tasks = []
            if channels_with_chain:
                all_chain_cids = list(
                    {cid for ch in channels_with_chain for cid in chain_by_ch[ch]}
                )
                tasks.append(
                    self.conn.search(
                        index=CHECKPOINT_WRITES_INDEX,
                        body={
                            "query": {
                                "bool": {
                                    "filter": [
                                        {"term": {"thread_id": thread_id}},
                                        {"term": {"checkpoint_ns": checkpoint_ns}},
                                        {"terms": {"channel": channels_with_chain}},
                                        {"terms": {"checkpoint_id": all_chain_cids}},
                                    ]
                                }
                            },
                            "size": 10000,
                        },
                    )
                )
            else:
                tasks.append(asyncio.sleep(0))  # placeholder

            if channels_with_seed:
                seed_ids = [
                    _make_blob_doc_id(
                        thread_id, checkpoint_ns, ch, cast(str, seed_ver_by_ch[ch])
                    )
                    for ch in channels_with_seed
                ]
                tasks.append(
                    self.conn.mget(
                        index=CHECKPOINT_BLOBS_INDEX, body={"ids": seed_ids}
                    )
                )
            else:
                tasks.append(asyncio.sleep(0))  # placeholder

            results = await asyncio.gather(*tasks)

            if channels_with_chain and results[0] is not None:
                ch_chain_set = {
                    ch: set(chain_by_ch[ch]) for ch in channels_with_chain
                }
                raw_hits = results[0]["hits"]["hits"]
                write_hits = [
                    h for h in raw_hits
                    if h["_source"]["checkpoint_id"]
                    in ch_chain_set.get(h["_source"]["channel"], set())
                ]

            if channels_with_seed and results[1] is not None:
                blob_docs = results[1]["docs"]

        return self._build_delta_channels_writes_history(
            channels=channels,
            chain_by_ch=chain_by_ch,
            seed_ver_by_ch=seed_ver_by_ch,
            write_hits=write_hits,
            blob_docs=blob_docs,
        )

    # ------------------------------------------------------------------
    # Sync shims (delegate to async via run_coroutine_threadsafe)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._check_not_event_loop()
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        self._check_not_event_loop()
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    aiter_.__anext__(),  # type: ignore[attr-defined]
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        self._check_not_event_loop()
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()

    def get_delta_channel_history(
        self,
        *,
        config: RunnableConfig,
        channels: Sequence[str],
    ) -> Mapping[str, DeltaChannelHistory]:
        self._check_not_event_loop()
        return asyncio.run_coroutine_threadsafe(
            self.aget_delta_channel_history(config=config, channels=channels),
            self.loop,
        ).result()

    def _check_not_event_loop(self) -> None:
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncElasticsearchSaver must be made from a "
                    "different thread. Use the async interface (aget_tuple, alist, etc.) "
                    "from within the event loop."
                )
        except RuntimeError:
            pass


__all__ = ["AsyncElasticsearchSaver"]
