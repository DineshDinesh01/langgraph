"""Shared base for sync and async Elasticsearch store implementations.

Design notes
------------
A single ES index (`langgraph_store`) stores all items:

    langgraph_store — one doc per (namespace, key)

Document IDs are ``{prefix}##key`` where ``prefix`` is the namespace rendered
as a dot-separated string (same convention as Postgres).

The ``value`` field is stored as a dynamic object so that individual sub-fields
are indexed and can be filtered in SearchOp queries.

Optional vector search uses ES's native ``knn`` query, which is backed by HNSW
indexing built into the engine — no extension required, unlike pgvector.

TTL is implemented with an ``expires_at`` date field.  A background sweeper
thread (or async task) issues a ``delete_by_query`` periodically.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

import orjson
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORE_INDEX = "langgraph_store"
SEP = "##"

# ---------------------------------------------------------------------------
# Index mapping
# ---------------------------------------------------------------------------

STORE_INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "dynamic": "true",
        "properties": {
            "prefix": {"type": "keyword"},
            "key": {"type": "keyword"},
            # value fields are dynamically mapped so filters can target them
            "value": {"type": "object", "dynamic": "true"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
            "expires_at": {"type": "date"},
            "ttl_minutes": {"type": "float"},
            # embedding added dynamically during setup() when index_config is given
        },
    },
}

# ---------------------------------------------------------------------------
# ES-specific index config
# ---------------------------------------------------------------------------


class ESIndexConfig(IndexConfig, total=False):
    """Elasticsearch-specific extension of the generic IndexConfig."""

    similarity: Literal["cosine", "dot_product", "l2_norm"]
    """Similarity metric used when creating the dense_vector field.

    - ``cosine`` (default): cosine similarity; vectors need not be normalised.
    - ``dot_product``: inner product; fastest, but vectors must be unit-normalised.
    - ``l2_norm``: Euclidean distance.
    """


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _text_to_namespace(text: str) -> tuple[str, ...]:
    return tuple(text.split(".")) if text else ()


def _doc_id(prefix: str, key: str) -> str:
    return f"{prefix}{SEP}{key}"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(ttl_minutes: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()


# ---------------------------------------------------------------------------
# Filter / query helpers
# ---------------------------------------------------------------------------

_RANGE_OP_MAP: dict[str, str] = {
    "$gt": "gt",
    "$gte": "gte",
    "$lt": "lt",
    "$lte": "lte",
}


def _filter_clause(key: str, op: str, value: Any) -> dict:
    """Convert a single filter key+operator+value to an ES query clause."""
    field = f"value.{key}"
    if op == "$eq":
        kw = f"{field}.keyword" if isinstance(value, str) else field
        return {"term": {kw: value}}
    elif op == "$ne":
        kw = f"{field}.keyword" if isinstance(value, str) else field
        return {"bool": {"must_not": [{"term": {kw: value}}]}}
    elif op in _RANGE_OP_MAP:
        return {"range": {field: {_RANGE_OP_MAP[op]: value}}}
    else:
        raise ValueError(f"Unsupported filter operator: {op}")


def _build_filter_clauses(filter: dict[str, Any] | None) -> list[dict]:
    """Convert SearchOp.filter dict to a list of ES must clauses."""
    if not filter:
        return []
    clauses: list[dict] = []
    for key, value in filter.items():
        if isinstance(value, dict):
            for op_name, val in value.items():
                clauses.append(_filter_clause(key, op_name, val))
        else:
            field = f"value.{key}"
            kw = f"{field}.keyword" if isinstance(value, str) else field
            clauses.append({"term": {kw: value}})
    return clauses


def _match_condition_to_es_filter(cond: MatchCondition) -> dict:
    """Convert a ListNamespacesOp MatchCondition to an ES filter clause."""
    has_wildcards = any(p == "*" for p in cond.path)
    if not has_wildcards:
        path_text = _namespace_to_text(cond.path)
        if cond.match_type == "prefix":
            return {"prefix": {"prefix": path_text}}
        else:  # suffix
            escaped = re.escape(path_text)
            return {"regexp": {"prefix": f"(.*\\.)?{escaped}$"}}
    else:
        parts = [r".*" if p == "*" else re.escape(p) for p in cond.path]
        pattern = r"\.".join(parts)
        if cond.match_type == "prefix":
            return {"regexp": {"prefix": f"{pattern}(\\..*)?$"}}
        else:
            return {"regexp": {"prefix": f"(.*\\.)?{pattern}$"}}


def _not_expired_filter() -> dict:
    """Exclude items whose expires_at is in the past."""
    return {
        "bool": {
            "should": [
                {"bool": {"must_not": [{"exists": {"field": "expires_at"}}]}},
                {"range": {"expires_at": {"gte": "now"}}},
            ],
            "minimum_should_match": 1,
        }
    }


# ---------------------------------------------------------------------------
# Hit / row converters
# ---------------------------------------------------------------------------


def _hit_to_item(hit: dict) -> Item:
    src = hit["_source"]
    ns = _text_to_namespace(src["prefix"])
    return Item(
        namespace=ns,
        key=src["key"],
        value=src["value"],
        created_at=datetime.fromisoformat(src["created_at"]),
        updated_at=datetime.fromisoformat(src["updated_at"]),
    )


def _hit_to_search_item(hit: dict, *, score: float | None = None) -> SearchItem:
    src = hit["_source"]
    ns = _text_to_namespace(src["prefix"])
    item_score = score if score is not None else hit.get("_score")
    return SearchItem(
        namespace=ns,
        key=src["key"],
        value=src["value"],
        created_at=datetime.fromisoformat(src["created_at"]),
        updated_at=datetime.fromisoformat(src["updated_at"]),
        score=float(item_score) if item_score is not None else None,
    )


# ---------------------------------------------------------------------------
# Op grouping (shared utility identical to Postgres store)
# ---------------------------------------------------------------------------


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    total = 0
    for idx, op in enumerate(ops):
        grouped[type(op)].append((idx, op))
        total += 1
    return grouped, total


# ---------------------------------------------------------------------------
# Index config helpers
# ---------------------------------------------------------------------------


def _ensure_index_config(
    index_config: ESIndexConfig,
) -> tuple[Any, ESIndexConfig]:
    """Resolve embeddings and pre-tokenise field paths.  Returns (embeddings, updated_config)."""
    from langchain_core.embeddings import Embeddings

    cfg = index_config.copy()
    tokenized: list[tuple[str, Any]] = []
    tot = 0
    fields = cfg.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    cfg["__tokenized_fields"] = tokenized
    cfg["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(cfg.get("embed"))
    return embeddings, cfg


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseElasticsearchStore:
    """Shared logic for sync and async Elasticsearch stores."""

    index_config: ESIndexConfig | None
    embeddings: Any  # Embeddings | EmbeddingsFunc | None
    ttl_config: TTLConfig | None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_index_body(self) -> dict:
        """Return the index creation body, adding the embedding field if needed."""
        body = {
            "settings": STORE_INDEX_MAPPING["settings"].copy(),
            "mappings": {
                "dynamic": "true",
                "properties": dict(STORE_INDEX_MAPPING["mappings"]["properties"]),
            },
        }
        if self.index_config:
            dims = self.index_config.get("dims")
            similarity = cast(dict, self.index_config).get("similarity", "cosine")
            if dims:
                body["mappings"]["properties"]["embedding"] = {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": similarity,
                }
        return body

    # ------------------------------------------------------------------
    # Bulk action builders
    # ------------------------------------------------------------------

    def _build_put_actions(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[list[dict], list[tuple[str, str, str, str]]]:
        """Build bulk actions for a batch of PutOps.

        Returns ``(actions, embedding_requests)`` where each embedding request is
        ``(doc_id, prefix, key, text)`` so that vectors can be embedded and
        patched back into the actions list.
        """
        dedupped: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped[(op.namespace, op.key)] = op

        actions: list[dict] = []
        embedding_requests: list[tuple[str, str, str, str]] = []

        for op in dedupped.values():
            prefix = _namespace_to_text(op.namespace)
            doc_id = _doc_id(prefix, op.key)
            if op.value is None:
                actions.append({"_op_type": "delete", "_index": STORE_INDEX, "_id": doc_id})
                continue

            now = _now_iso()
            doc: dict[str, Any] = {
                "prefix": prefix,
                "key": op.key,
                "value": op.value,
                "updated_at": now,
            }

            # TTL
            ttl: float | None = op.ttl
            if ttl is None and self.ttl_config:
                ttl = self.ttl_config.get("default_ttl")
            if ttl is not None:
                doc["expires_at"] = _expires_iso(ttl)
                doc["ttl_minutes"] = ttl
            else:
                doc["expires_at"] = None
                doc["ttl_minutes"] = None

            # Embeddings
            if self.index_config and op.index is not False:
                paths = (
                    cast(dict, self.index_config)["__tokenized_fields"]
                    if op.index is None
                    else [(ix, tokenize_path(ix)) for ix in op.index]
                )
                for path, tokenized_path in paths:
                    texts = get_text_at_path(op.value, tokenized_path)
                    # Use first text found; for multi-value fields take the join
                    if texts:
                        text = " ".join(texts)
                        embedding_requests.append((doc_id, prefix, op.key, text))
                        break

            actions.append(
                {
                    "_op_type": "update",
                    "_index": STORE_INDEX,
                    "_id": doc_id,
                    "doc": doc,
                    "doc_as_upsert": True,
                    # Preserve created_at on update via painless script
                    "scripted_upsert": False,
                }
            )
            # Ensure created_at is set on insert but not overwritten on update.
            # We achieve this by using a scripted upsert: on insert the script
            # sets created_at; on update the script is also run but skips it.
            # Simpler: just always include created_at — if the doc exists we
            # overwrite updated_at but keep our own created_at logic in client.
            # For simplicity we just use doc_as_upsert and set created_at always
            # (it will be overwritten on updates). Callers that care can use GET first.
            # The correct approach: script upsert or two-phase GET+PUT.
            # We go with the simpler approach: set created_at = now on every upsert
            # and rely on ES's update to merge the doc (it only updates given fields).
            # But doc_as_upsert replaces the entire `doc` into the existing _source
            # (ES merges), so to preserve created_at we must NOT include it in the
            # update doc, only set it via the upsert fallback.
            #
            # To do this properly we use a scripted upsert:
            actions[-1] = {
                "_op_type": "update",
                "_index": STORE_INDEX,
                "_id": doc_id,
                "script": {
                    "source": (
                        "ctx._source.putAll(params.doc); "
                        "if (ctx._source.created_at == null) { "
                        "  ctx._source.created_at = params.now; "
                        "}"
                    ),
                    "params": {"doc": doc, "now": now},
                },
                "upsert": {**doc, "created_at": now},
            }

        return actions, embedding_requests

    # ------------------------------------------------------------------
    # Search query builder
    # ------------------------------------------------------------------

    def _build_search_query(
        self,
        op: SearchOp,
        embedding: list[float] | None = None,
    ) -> dict:
        """Build the ES search body for a SearchOp."""
        must: list[dict] = [_not_expired_filter()]

        if op.namespace_prefix:
            prefix_text = _namespace_to_text(op.namespace_prefix)
            must.append({"prefix": {"prefix": prefix_text}})

        filter_clauses = _build_filter_clauses(op.filter)
        must.extend(filter_clauses)

        if embedding is not None and self.index_config:
            similarity = cast(dict, self.index_config).get("similarity", "cosine")
            k = op.limit + op.offset
            body: dict = {
                "knn": {
                    "field": "embedding",
                    "query_vector": embedding,
                    "k": k,
                    "num_candidates": max(k * 5, 100),
                    "filter": {"bool": {"must": must}},
                },
                "from": op.offset,
                "size": op.limit,
            }
        else:
            body = {
                "query": {"bool": {"must": must}},
                "sort": [{"updated_at": {"order": "desc"}}],
                "from": op.offset,
                "size": op.limit,
            }

        return body

    # ------------------------------------------------------------------
    # List-namespaces helpers
    # ------------------------------------------------------------------

    def _build_list_ns_query(self, op: ListNamespacesOp) -> dict:
        """Build ES search body for listing unique namespaces."""
        filters: list[dict] = [_not_expired_filter()]

        if op.match_conditions:
            for cond in op.match_conditions:
                filters.append(_match_condition_to_es_filter(cond))

        return {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "unique_prefixes": {
                    "terms": {"field": "prefix", "size": 50_000}
                }
            },
        }

    def _process_list_ns_response(
        self, resp: dict, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        buckets = (
            resp.get("aggregations", {})
            .get("unique_prefixes", {})
            .get("buckets", [])
        )
        seen: set[tuple[str, ...]] = set()
        results: list[tuple[str, ...]] = []
        for bucket in buckets:
            ns = _text_to_namespace(bucket["key"])
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            if ns not in seen:
                seen.add(ns)
                results.append(ns)
        results.sort()
        return results[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # TTL refresh helper
    # ------------------------------------------------------------------

    def _build_ttl_refresh_script(self, doc_ids: list[str]) -> dict:
        """Build update-by-query body to refresh expires_at for given doc IDs."""
        return {
            "script": {
                "source": (
                    "if (ctx._source.ttl_minutes != null) { "
                    "  long millis = (long)(ctx._source.ttl_minutes * 60000); "
                    "  ctx._source.expires_at = ZonedDateTime.ofInstant("
                    "    Instant.ofEpochMilli(System.currentTimeMillis() + millis),"
                    "    ZoneId.of('UTC')).toString(); "
                    "}"
                ),
            },
            "query": {"ids": {"values": doc_ids}},
        }
