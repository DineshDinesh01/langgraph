"""Shared base for sync and async Elasticsearch checkpoint savers.

Design notes
------------
Three ES indices mirror the three Postgres tables:

    langgraph_checkpoints       — one doc per (thread_id, checkpoint_ns, checkpoint_id)
    langgraph_checkpoint_blobs  — one doc per (thread_id, checkpoint_ns, channel, version)
    langgraph_checkpoint_writes — one doc per (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)

Document IDs are built by joining the primary-key fields with SEP ("##").
SEP is chosen to be unlikely in user-supplied thread IDs (which are often UUIDs).

Binary data (BYTEA in Postgres) is stored as base64-encoded strings in the
ES `binary` field type, which ES itself encodes/decodes as base64.

Primitive channel values (str, int, float, bool, None) are stored inline in the
checkpoint JSON document. Non-primitive values are serialised with `serde` and
stored in langgraph_checkpoint_blobs; their slot in checkpoint.channel_values
is replaced with the boolean True as a marker.

The delta-channel history walk (stage 1 + stage 2) is a direct port of the
pure-Python logic from checkpoint-postgres/base.py.  Only the DB I/O differs.
"""

from __future__ import annotations

import base64
import random
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointTuple,
    DeltaChannelHistory,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS

# Page size for the stage-1 paged scan in get_delta_channel_history.
_DELTA_PAGE_SIZE = 1024

# Separator used to build composite document IDs.
# Must not appear in thread_id, checkpoint_ns, checkpoint_id, channel, or version.
SEP = "##"

# ---------------------------------------------------------------------------
# Index names
# ---------------------------------------------------------------------------

CHECKPOINT_INDEX = "langgraph_checkpoints"
CHECKPOINT_BLOBS_INDEX = "langgraph_checkpoint_blobs"
CHECKPOINT_WRITES_INDEX = "langgraph_checkpoint_writes"

# ---------------------------------------------------------------------------
# Index mappings  (ES equivalent of CREATE TABLE)
# ---------------------------------------------------------------------------

CHECKPOINT_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "thread_id":            {"type": "keyword"},
            "checkpoint_ns":        {"type": "keyword"},
            "checkpoint_id":        {"type": "keyword"},
            "parent_checkpoint_id": {"type": "keyword"},
            "type":                 {"type": "keyword"},
            # Store the full checkpoint JSON but don't index its internals —
            # we never filter by field values inside checkpoint.
            "checkpoint":           {"type": "object", "enabled": False},
            # metadata IS queried (filter by metadata key/value), so leave enabled.
            "metadata":             {"type": "object"},
        }
    }
}

CHECKPOINT_BLOBS_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "thread_id":     {"type": "keyword"},
            "checkpoint_ns": {"type": "keyword"},
            "channel":       {"type": "keyword"},
            "version":       {"type": "keyword"},
            "type":          {"type": "keyword"},
            # ES binary field stores base64-encoded bytes.
            "blob":          {"type": "binary"},
        }
    }
}

CHECKPOINT_WRITES_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "thread_id":     {"type": "keyword"},
            "checkpoint_ns": {"type": "keyword"},
            "checkpoint_id": {"type": "keyword"},
            "task_id":       {"type": "keyword"},
            "task_path":     {"type": "keyword"},
            "idx":           {"type": "integer"},
            "channel":       {"type": "keyword"},
            "type":          {"type": "keyword"},
            "blob":          {"type": "binary"},
        }
    }
}

# ---------------------------------------------------------------------------
# Document ID helpers
# ---------------------------------------------------------------------------


def _make_checkpoint_doc_id(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return SEP.join([thread_id, checkpoint_ns, checkpoint_id])


def _make_blob_doc_id(thread_id: str, checkpoint_ns: str, channel: str, version: str) -> str:
    return SEP.join([thread_id, checkpoint_ns, channel, version])


def _make_write_doc_id(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
) -> str:
    return SEP.join([thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)])


# ---------------------------------------------------------------------------
# Blob encode / decode helpers
# ---------------------------------------------------------------------------


def _encode_blob(blob: bytes | None) -> str | None:
    """Encode raw bytes to a base64 string for ES binary field storage."""
    if blob is None:
        return None
    return base64.b64encode(blob).decode("ascii")


def _decode_blob(b64: str | None) -> bytes | None:
    """Decode a base64 string from ES binary field back to raw bytes."""
    if b64 is None:
        return None
    return base64.b64decode(b64)


# ---------------------------------------------------------------------------
# BaseElasticsearchSaver
# ---------------------------------------------------------------------------


class BaseElasticsearchSaver(BaseCheckpointSaver[str]):
    """Shared logic for ElasticsearchSaver and AsyncElasticsearchSaver.

    Subclasses provide the actual ES I/O (sync or async); this class contains:
    - index/mapping constants
    - serialization helpers (_dump_blobs, _load_blobs, _dump_writes, _load_writes)
    - CheckpointTuple assembly (_load_checkpoint_tuple)
    - ES filter builder (_search_filters)
    - version generator (get_next_version)
    - pure-Python delta-channel walk logic (ported from checkpoint-postgres)
    """

    CHECKPOINT_INDEX = CHECKPOINT_INDEX
    CHECKPOINT_BLOBS_INDEX = CHECKPOINT_BLOBS_INDEX
    CHECKPOINT_WRITES_INDEX = CHECKPOINT_WRITES_INDEX

    # ------------------------------------------------------------------
    # Version generator (identical to BasePostgresSaver)
    # ------------------------------------------------------------------

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Return a new monotonically-increasing version string.

        Format: "{counter:032}.{random_float:016}"
        Sortable lexicographically, unique enough for practical use.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[dict[str, Any]]:
        """Build ES bulk-index actions for channel blobs.

        Only called for channels whose values are non-primitive (i.e. stored
        separately from the checkpoint JSON document).

        Each action uses _op_type="create" (insert-only) because blobs are
        content-addressed: the same (channel, version) always has the same
        content, so silently skipping a duplicate is correct.
        """
        if not versions:
            return []

        actions = []
        for channel, version in versions.items():
            doc_id = _make_blob_doc_id(thread_id, checkpoint_ns, channel, str(version))
            if channel in values:
                type_tag, blob = self.serde.dumps_typed(values[channel])
                blob_b64 = _encode_blob(blob)
            else:
                type_tag, blob_b64 = "empty", None

            actions.append({
                "_index": CHECKPOINT_BLOBS_INDEX,
                "_id": doc_id,
                "_op_type": "create",  # ON CONFLICT DO NOTHING equivalent
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "channel": channel,
                "version": str(version),
                "type": type_tag,
                "blob": blob_b64,
            })
        return actions

    def _load_blobs(self, mget_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Deserialise channel blobs from an ES mget response's `docs` list."""
        result: dict[str, Any] = {}
        for doc in mget_docs:
            if not doc.get("found"):
                continue
            src = doc["_source"]
            type_tag: str = src["type"]
            if type_tag == "empty":
                continue
            blob = _decode_blob(src.get("blob"))
            if blob is None:
                continue
            channel: str = src["channel"]
            result[channel] = self.serde.loads_typed((type_tag, blob))
        return result

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build ES bulk actions for intermediate task writes.

        Special channels (in WRITES_IDX_MAP) use _op_type="index" (upsert)
        because they are overwritten on retry.  All other channels use
        _op_type="create" (insert-only) to avoid clobbering completed writes.
        """
        actions = []
        for idx, (channel, value) in enumerate(writes):
            real_idx = WRITES_IDX_MAP.get(channel, idx)
            doc_id = _make_write_doc_id(
                thread_id, checkpoint_ns, checkpoint_id, task_id, real_idx
            )
            type_tag, blob = self.serde.dumps_typed(value)
            blob_b64 = _encode_blob(blob)
            op_type = "index" if channel in WRITES_IDX_MAP else "create"
            actions.append({
                "_index": CHECKPOINT_WRITES_INDEX,
                "_id": doc_id,
                "_op_type": op_type,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": real_idx,
                "channel": channel,
                "type": type_tag,
                "blob": blob_b64,
            })
        return actions

    def _load_writes(self, hits: list[dict[str, Any]]) -> list[PendingWrite]:
        """Deserialise pending writes from ES search hits.

        Returns list of (task_id, channel, value) tuples, sorted by
        task_id then idx (same order as Postgres ORDER BY cw.task_id, cw.idx).
        """
        if not hits:
            return []
        rows = []
        for hit in hits:
            src = hit["_source"]
            type_tag: str = src["type"]
            blob = _decode_blob(src.get("blob"))
            value = self.serde.loads_typed((type_tag, blob if blob is not None else b""))
            rows.append((src["task_id"], src["idx"], src["channel"], value))
        rows.sort(key=lambda r: (r[0], r[1]))
        return [(task_id, channel, value) for task_id, _idx, channel, value in rows]

    def _migrate_pending_sends(
        self,
        pending_sends_hits: list[dict[str, Any]],
        checkpoint: dict[str, Any],
        channel_values: dict[str, Any],
    ) -> None:
        """Back-fill pending_sends for old checkpoint format (v < 4)."""
        if not pending_sends_hits:
            return
        sends: list[tuple[bytes, bytes]] = []
        for hit in sorted(
            pending_sends_hits,
            key=lambda h: (
                h["_source"].get("task_path", ""),
                h["_source"]["task_id"],
                h["_source"]["idx"],
            ),
        ):
            src = hit["_source"]
            type_b = src["type"].encode()
            blob = _decode_blob(src.get("blob")) or b""
            sends.append((type_b, blob))

        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c.decode(), b)) for c, b in sends],
        )
        channel_values[TASKS] = self.serde.loads_typed((enc, blob))
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    # ------------------------------------------------------------------
    # CheckpointTuple assembly
    # ------------------------------------------------------------------

    def _load_checkpoint_tuple(
        self,
        checkpoint_doc: dict[str, Any],
        blob_docs: list[dict[str, Any]],
        write_hits: list[dict[str, Any]],
    ) -> CheckpointTuple:
        """Assemble a CheckpointTuple from three separate ES responses.

        checkpoint_doc — a single ES _source dict from the checkpoints index.
        blob_docs      — the `docs` list from an mget call on checkpoint_blobs.
        write_hits     — the `hits.hits` list from a search on checkpoint_writes.
        """
        src = checkpoint_doc["_source"]
        thread_id: str = src["thread_id"]
        checkpoint_ns: str = src["checkpoint_ns"]
        checkpoint_id: str = src["checkpoint_id"]
        parent_checkpoint_id: str | None = src.get("parent_checkpoint_id")

        checkpoint_data: dict[str, Any] = {
            **src["checkpoint"],
            "channel_values": {
                **(src["checkpoint"].get("channel_values") or {}),
                **self._load_blobs(blob_docs),
            },
        }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint_data,
            metadata=src.get("metadata", {}),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=self._load_writes(write_hits),
        )

    # ------------------------------------------------------------------
    # ES filter builder  (replaces _search_where from Postgres)
    # ------------------------------------------------------------------

    def _search_filters(
        self,
        config: RunnableConfig | None,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of ES filter clauses for list() / alist().

        Equivalent to BasePostgresSaver._search_where() but returns ES
        filter dicts instead of SQL fragments.
        """
        clauses: list[dict[str, Any]] = []

        if config:
            clauses.append({"term": {"thread_id": config["configurable"]["thread_id"]}})
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                clauses.append({"term": {"checkpoint_ns": checkpoint_ns}})
            if checkpoint_id := get_checkpoint_id(config):
                clauses.append({"term": {"checkpoint_id": checkpoint_id}})

        # metadata containment filter — Postgres uses @>
        # In ES, each key becomes a separate term filter on metadata.<key>
        if filter:
            for key, value in filter.items():
                clauses.append({"term": {f"metadata.{key}": value}})

        if before is not None:
            clauses.append(
                {"range": {"checkpoint_id": {"lt": get_checkpoint_id(before)}}}
            )

        return clauses

    # ------------------------------------------------------------------
    # Blob ID list for a checkpoint  (used by get_tuple)
    # ------------------------------------------------------------------

    def _blob_ids_for_checkpoint(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_src: dict[str, Any],
    ) -> list[str]:
        """Return the list of blob document IDs needed for this checkpoint.

        Only channels whose stored channel_values entry is True (the blob
        marker set during put()) actually have a blob document — primitive
        values are stored inline.
        """
        channel_versions: dict[str, str] = (
            checkpoint_src.get("checkpoint", {}).get("channel_versions") or {}
        )
        stored_values: dict[str, Any] = (
            checkpoint_src.get("checkpoint", {}).get("channel_values") or {}
        )
        return [
            _make_blob_doc_id(thread_id, checkpoint_ns, ch, str(ver))
            for ch, ver in channel_versions.items()
            if stored_values.get(ch) is True
        ]

    # ------------------------------------------------------------------
    # Delta channel history — pure Python walk logic
    # (ported directly from checkpoint-postgres/base.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _ingest_stage1_page(
        stage1_rows: Sequence[Mapping[str, Any]],
        channels: Sequence[str],
        parent_of: dict[str, str | None],
        ver_by_i_by_cid: list[dict[str, str | None]],
        hs_by_i_by_cid: list[dict[str, bool]],
    ) -> str | None:
        """Fold one stage-1 page into the running walk-state mappings.

        Each row must have: checkpoint_id, parent_checkpoint_id,
        channel_versions (dict), channel_values (dict).

        Returns the oldest checkpoint_id seen (used as next-page cursor).
        """
        oldest: str | None = None
        for r in stage1_rows:
            cid = cast(str, r["checkpoint_id"])
            parent_of[cid] = cast("str | None", r.get("parent_checkpoint_id"))
            channel_versions: dict[str, Any] = r.get("channel_versions") or {}
            channel_values: dict[str, Any] = r.get("channel_values") or {}
            for i, ch in enumerate(channels):
                ver_by_i_by_cid[i][cid] = channel_versions.get(ch)
                hs_by_i_by_cid[i][cid] = channel_values.get(ch) is not None
            oldest = cid
        return oldest

    @staticmethod
    def _try_advance_walks(
        target_id: str,
        channels: Sequence[str],
        parent_of: Mapping[str, str | None],
        ver_by_i_by_cid: Sequence[Mapping[str, str | None]],
        hs_by_i_by_cid: Sequence[Mapping[str, bool]],
        chain_by_ch: dict[str, list[str]],
        seed_ver_by_ch: dict[str, str | None],
        walk_cursor_by_ch: dict[str, str | None],
        seeded: set[str],
    ) -> None:
        """Advance each not-yet-seeded channel's walk as far as possible."""
        for i, ch in enumerate(channels):
            if ch in seeded:
                continue
            if ch not in walk_cursor_by_ch:
                walk_cursor_by_ch[ch] = parent_of.get(target_id)
            cur_cid = walk_cursor_by_ch[ch]
            ch_chain = chain_by_ch[ch]
            hs_i = hs_by_i_by_cid[i]
            ver_i = ver_by_i_by_cid[i]
            while cur_cid is not None:
                if cur_cid not in parent_of:
                    break
                ch_chain.append(cur_cid)
                if hs_i.get(cur_cid, False):
                    seed_ver_by_ch[ch] = ver_i.get(cur_cid)
                    seeded.add(ch)
                    cur_cid = None
                    break
                cur_cid = parent_of[cur_cid]
            walk_cursor_by_ch[ch] = cur_cid

    def _build_delta_channels_writes_history(
        self,
        *,
        channels: Sequence[str],
        chain_by_ch: Mapping[str, list[str]],
        seed_ver_by_ch: Mapping[str, str | None],
        write_hits: list[dict[str, Any]],
        blob_docs: list[dict[str, Any]],
    ) -> dict[str, DeltaChannelHistory]:
        """Assemble per-channel DeltaChannelHistory from stage-2 ES results."""
        # Group writes by (channel, checkpoint_id)
        writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes | None, str, int]]]] = {
            ch: {} for ch in channels
        }
        for hit in write_hits:
            src = hit["_source"]
            ch: str = src["channel"]
            cid: str = src["checkpoint_id"]
            type_tag: str = src["type"]
            blob = _decode_blob(src.get("blob"))
            task_id: str = src["task_id"]
            idx: int = src["idx"]
            writes_by_ch_by_cid.setdefault(ch, {}).setdefault(cid, []).append(
                (type_tag, blob, task_id, idx)
            )

        # Sort writes per (channel, cid) by (task_id, idx) descending
        for cid_map in writes_by_ch_by_cid.values():
            for ws in cid_map.values():
                ws.sort(key=lambda w: (w[2], w[3]), reverse=True)

        # Index seed blobs by (channel, version)
        seed_blob_by_ver: dict[tuple[str, str], tuple[str, bytes | None]] = {}
        for doc in blob_docs:
            if not doc.get("found"):
                continue
            src = doc["_source"]
            ch = src["channel"]
            ver: str = src["version"]
            seed_blob_by_ver[(ch, ver)] = (src["type"], _decode_blob(src.get("blob")))

        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            chain_cids = chain_by_ch.get(ch, [])
            seed_version = seed_ver_by_ch.get(ch)

            collected: list[PendingWrite] = []
            cid_writes = writes_by_ch_by_cid.get(ch, {})
            for cid in chain_cids:
                for type_tag, blob, task_id, _idx in cid_writes.get(cid, []):
                    val = self.serde.loads_typed((type_tag, blob if blob is not None else b""))
                    collected.append((task_id, ch, val))
            collected.reverse()

            entry: DeltaChannelHistory = {"writes": collected}
            if seed_version is not None:
                blob_entry = seed_blob_by_ver.get((ch, seed_version))
                if blob_entry is not None and blob_entry[0] != "empty":
                    entry["seed"] = self.serde.loads_typed(
                        (blob_entry[0], blob_entry[1] if blob_entry[1] is not None else b"")
                    )
            result[ch] = entry
        return result

    # ------------------------------------------------------------------
    # Stage-1 ES query builder  (replaces _build_delta_stage1_sql)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_delta_stage1_query(
        thread_id: str,
        checkpoint_ns: str,
        cursor: str | None,
    ) -> dict[str, Any]:
        """Build the ES search body for stage-1 of get_delta_channel_history.

        Fetches checkpoint_id, parent_checkpoint_id, channel_versions and
        channel_values (as source filters) ordered newest-first with an
        optional cursor for paging.

        The response rows are fed into _ingest_stage1_page.
        """
        filters: list[dict[str, Any]] = [
            {"term": {"thread_id": thread_id}},
            {"term": {"checkpoint_ns": checkpoint_ns}},
        ]
        if cursor is not None:
            filters.append({"range": {"checkpoint_id": {"lt": cursor}}})

        return {
            "query": {"bool": {"filter": filters}},
            "_source": [
                "checkpoint_id",
                "parent_checkpoint_id",
                "checkpoint.channel_versions",
                "checkpoint.channel_values",
            ],
            "sort": [{"checkpoint_id": "desc"}],
            "size": _DELTA_PAGE_SIZE,
        }

    @staticmethod
    def _flatten_stage1_hit(hit: dict[str, Any]) -> dict[str, Any]:
        """Flatten an ES stage-1 hit into the row shape expected by _ingest_stage1_page."""
        src = hit["_source"]
        checkpoint = src.get("checkpoint") or {}
        return {
            "checkpoint_id": src["checkpoint_id"],
            "parent_checkpoint_id": src.get("parent_checkpoint_id"),
            "channel_versions": checkpoint.get("channel_versions") or {},
            "channel_values": checkpoint.get("channel_values") or {},
        }
