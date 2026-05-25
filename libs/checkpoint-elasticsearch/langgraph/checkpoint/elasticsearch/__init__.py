"""Synchronous Elasticsearch checkpoint saver for LangGraph."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
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

from langgraph.checkpoint.elasticsearch._internal import Conn
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


class ElasticsearchSaver(BaseElasticsearchSaver):
    """Checkpointer that stores LangGraph checkpoints in Elasticsearch.

    Usage::

        from langgraph.checkpoint.elasticsearch import ElasticsearchSaver

        with ElasticsearchSaver.from_conn_string("http://localhost:9200") as saver:
            saver.setup()
            graph = compiled_graph.compile(checkpointer=saver)
            result = graph.invoke(inputs, {"configurable": {"thread_id": "t1"}})

    The saver creates three indices on first use (via setup()):
        - langgraph_checkpoints
        - langgraph_checkpoint_blobs
        - langgraph_checkpoint_writes
    """

    def __init__(
        self,
        conn: Conn,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        hosts: str | list[str],
        *,
        serde: SerializerProtocol | None = None,
        **es_kwargs: Any,
    ) -> Iterator[ElasticsearchSaver]:
        """Create an ElasticsearchSaver from a host string or list.

        Args:
            hosts: ES host URL(s), e.g. "http://localhost:9200".
            serde: Optional custom serializer.
            **es_kwargs: Extra kwargs forwarded to the Elasticsearch client.

        Example::

            with ElasticsearchSaver.from_conn_string("http://localhost:9200") as saver:
                saver.setup()
        """
        client = Elasticsearch(hosts, **es_kwargs)
        try:
            yield cls(client, serde=serde)
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create the three ES indices if they do not already exist.

        Must be called once before first use.  Safe to call again — existing
        indices are left unchanged (ignore_400 suppresses "already exists").
        """
        for index, mapping in [
            (CHECKPOINT_INDEX, CHECKPOINT_INDEX_MAPPING),
            (CHECKPOINT_BLOBS_INDEX, CHECKPOINT_BLOBS_INDEX_MAPPING),
            (CHECKPOINT_WRITES_INDEX, CHECKPOINT_WRITES_INDEX_MAPPING),
        ]:
            if not self.conn.indices.exists(index=index):
                self.conn.indices.create(index=index, body=mapping)

    # ------------------------------------------------------------------
    # get_tuple
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from Elasticsearch.

        If config contains a checkpoint_id, that specific checkpoint is
        returned.  Otherwise, the latest checkpoint for the thread is returned.

        Args:
            config: RunnableConfig with at least {"configurable": {"thread_id": ...}}.

        Returns:
            CheckpointTuple or None if no matching checkpoint exists.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.lock:
            # --- Step 1: fetch the checkpoint document ---
            if checkpoint_id:
                doc_id = _make_checkpoint_doc_id(thread_id, checkpoint_ns, checkpoint_id)
                resp = self.conn.get(index=CHECKPOINT_INDEX, id=doc_id, ignore=[404])
                if not resp.get("found"):
                    return None
                checkpoint_doc = resp
            else:
                resp = self.conn.search(
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

            # --- Step 2: fetch blobs via mget ---
            blob_ids = self._blob_ids_for_checkpoint(thread_id, checkpoint_ns, src)
            if blob_ids:
                mget_resp = self.conn.mget(
                    index=CHECKPOINT_BLOBS_INDEX,
                    body={"ids": blob_ids},
                )
                blob_docs = mget_resp["docs"]
            else:
                blob_docs = []

            # --- Step 3: fetch pending writes ---
            writes_resp = self.conn.search(
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

            # --- Step 4: back-fill pending_sends for old checkpoint format (v<4) ---
            checkpoint_data = src.get("checkpoint", {})
            if checkpoint_data.get("v", 4) < 4 and parent_checkpoint_id:
                sends_resp = self.conn.search(
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
    # list
    # ------------------------------------------------------------------

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the given criteria, newest first.

        Args:
            config: Filter by thread_id and optionally checkpoint_ns.
            filter: Filter by metadata key/value pairs.
            before: Only return checkpoints older than this config's checkpoint_id.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple for each matching checkpoint.
        """
        clauses = self._search_filters(config, filter, before)
        query = {"bool": {"filter": clauses}} if clauses else {"match_all": {}}
        body: dict[str, Any] = {
            "query": query,
            "sort": [{"checkpoint_id": "desc"}],
            "size": limit if limit is not None else 100,
        }

        with self.lock:
            resp = self.conn.search(index=CHECKPOINT_INDEX, body=body)
            checkpoint_docs = resp["hits"]["hits"]
            if not checkpoint_docs:
                return

            # Batch-fetch blobs and writes for all checkpoints in one pass.
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
                mget_resp = self.conn.mget(
                    index=CHECKPOINT_BLOBS_INDEX,
                    body={"ids": all_blob_ids},
                )
                for blob_doc in mget_resp["docs"]:
                    if blob_doc.get("found"):
                        blob_map[blob_doc["_id"]] = blob_doc

            checkpoint_ids = [d["_source"]["checkpoint_id"] for d in checkpoint_docs]
            writes_resp = self.conn.search(
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
    # put
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Elasticsearch.

        Primitive channel values (str, int, float, bool, None) are stored
        inline in the checkpoint document.  All other values are stored in
        langgraph_checkpoint_blobs and replaced with True in the document.

        Args:
            config: The associated runnable config.
            checkpoint: The checkpoint state to store.
            metadata: Additional metadata to store alongside.
            new_versions: Mapping of channel name → new version string.

        Returns:
            Updated RunnableConfig with the new checkpoint_id set.
        """
        configurable = config["configurable"].copy()
        thread_id: str = configurable.pop("thread_id")
        checkpoint_ns: str = configurable.pop("checkpoint_ns")
        parent_checkpoint_id: str | None = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        # Split channel values: primitives stay inline, others go to blobs.
        blob_values: dict[str, Any] = {}
        for k, v in checkpoint["channel_values"].items():
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy["channel_values"].pop(k)
                copy["channel_values"][k] = True
            elif v is None or isinstance(v, (str, int, float, bool)):
                pass  # store inline
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}

        with self.lock:
            # Write blobs first (they are content-addressed, safe to write before checkpoint).
            if blob_versions:
                actions = self._dump_blobs(
                    thread_id, checkpoint_ns, blob_values, blob_versions
                )
                bulk(self.conn, actions, raise_on_error=False)

            # Upsert the checkpoint document.
            doc_id = _make_checkpoint_doc_id(
                thread_id, checkpoint_ns, checkpoint["id"]
            )
            self.conn.index(
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
    # put_writes
    # ------------------------------------------------------------------

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate task writes linked to a checkpoint.

        Args:
            config: Config of the associated checkpoint.
            writes: List of (channel, value) pairs to store.
            task_id: Identifier of the task producing these writes.
            task_path: Optional path prefix for the task.
        """
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
        with self.lock:
            bulk(self.conn, actions, raise_on_error=False)

    # ------------------------------------------------------------------
    # delete_thread
    # ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, blobs, and writes for a thread.

        Args:
            thread_id: The thread whose data should be removed.
        """
        query = {"query": {"term": {"thread_id": thread_id}}}
        with self.lock:
            for index in [
                CHECKPOINT_INDEX,
                CHECKPOINT_BLOBS_INDEX,
                CHECKPOINT_WRITES_INDEX,
            ]:
                self.conn.delete_by_query(index=index, body=query, refresh=True)

    # ------------------------------------------------------------------
    # get_delta_channel_history  (two-stage walk)
    # ------------------------------------------------------------------

    def get_delta_channel_history(
        self,
        *,
        config: RunnableConfig,
        channels: Sequence[str],
    ) -> Mapping[str, DeltaChannelHistory]:
        """Retrieve the write history for specific channels up to a checkpoint.

        Uses a two-stage approach:
        - Stage 1: paged search over checkpoints index walking the parent chain
          to identify the write chain and seed version for each channel.
        - Stage 2: fetch the actual write blobs and seed blobs.

        Args:
            config: Config identifying the target checkpoint.
            channels: Channel names to reconstruct history for.

        Returns:
            Dict mapping channel name to DeltaChannelHistory.
        """
        if not channels:
            return {}

        channels = list(channels)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.lock:
            if checkpoint_id is None:
                target = self.get_tuple(config)
                if target is None:
                    return {ch: {"writes": []} for ch in channels}
                checkpoint_id = target.config["configurable"]["checkpoint_id"]

            # --- Stage 1: paged parent-chain walk ---
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
                resp = self.conn.search(index=CHECKPOINT_INDEX, body=stage1_body)
                page_hits = resp["hits"]["hits"]
                if not page_hits:
                    break

                rows = [self._flatten_stage1_hit(h) for h in page_hits]
                oldest = self._ingest_stage1_page(
                    rows,
                    channels,
                    parent_of,
                    ver_by_i_by_cid,
                    hs_by_i_by_cid,
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

            # --- Stage 2: fetch writes + seed blobs ---
            channels_with_chain = [ch for ch in channels if chain_by_ch[ch]]
            channels_with_seed = [
                ch for ch in channels if seed_ver_by_ch.get(ch) is not None
            ]

            # Fetch writes: all channels with non-empty chains in one query.
            write_hits: list[dict[str, Any]] = []
            if channels_with_chain:
                all_chain_cids = list(
                    {cid for ch in channels_with_chain for cid in chain_by_ch[ch]}
                )
                writes_resp = self.conn.search(
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
                raw_hits = writes_resp["hits"]["hits"]
                # Filter each hit to only include checkpoint_ids in that channel's chain.
                ch_chain_set = {ch: set(chain_by_ch[ch]) for ch in channels_with_chain}
                write_hits = [
                    h for h in raw_hits
                    if h["_source"]["checkpoint_id"]
                    in ch_chain_set.get(h["_source"]["channel"], set())
                ]

            # Fetch seed blobs via mget.
            blob_docs: list[dict[str, Any]] = []
            if channels_with_seed:
                seed_ids = [
                    _make_blob_doc_id(
                        thread_id, checkpoint_ns, ch, cast(str, seed_ver_by_ch[ch])
                    )
                    for ch in channels_with_seed
                ]
                mget_resp = self.conn.mget(
                    index=CHECKPOINT_BLOBS_INDEX,
                    body={"ids": seed_ids},
                )
                blob_docs = mget_resp["docs"]

        return self._build_delta_channels_writes_history(
            channels=channels,
            chain_by_ch=chain_by_ch,
            seed_ver_by_ch=seed_ver_by_ch,
            write_hits=write_hits,
            blob_docs=blob_docs,
        )


__all__ = ["ElasticsearchSaver", "BaseElasticsearchSaver"]
