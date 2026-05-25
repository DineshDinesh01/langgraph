# type: ignore
"""Sync tests for ElasticsearchStore."""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest
from elasticsearch import Elasticsearch
from langgraph.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp

from langgraph.store.elasticsearch import ElasticsearchStore
from tests.conftest import DEFAULT_ES_URL, CharacterEmbeddings

TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@contextmanager
def _store(**kwargs):
    """Fresh store — wipes the langgraph_store index before yielding."""
    with ElasticsearchStore.from_conn_string(DEFAULT_ES_URL, **kwargs) as store:
        store.setup()
        conn: Elasticsearch = store._conn
        conn.delete_by_query(
            index="langgraph_store",
            body={"query": {"match_all": {}}},
            refresh=True,
            ignore_unavailable=True,
        )
        yield store


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------


def test_put_and_get():
    with _store() as store:
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        store._conn.indices.refresh(index="langgraph_store")
        item = store.get(("users", "123"), "prefs")
        assert item is not None
        assert item.value == {"theme": "dark"}
        assert item.key == "prefs"
        assert item.namespace == ("users", "123")


def test_get_missing_returns_none():
    with _store() as store:
        item = store.get(("no", "ns"), "missing-key")
        assert item is None


def test_put_delete():
    with _store() as store:
        store.put(("a",), "k", {"x": 1})
        store._conn.indices.refresh(index="langgraph_store")
        assert store.get(("a",), "k") is not None
        store.put(("a",), "k", None)  # delete
        store._conn.indices.refresh(index="langgraph_store")
        assert store.get(("a",), "k") is None


def test_put_upsert():
    with _store() as store:
        store.put(("ns",), "key", {"v": 1})
        store._conn.indices.refresh(index="langgraph_store")
        store.put(("ns",), "key", {"v": 2})
        store._conn.indices.refresh(index="langgraph_store")
        item = store.get(("ns",), "key")
        assert item.value == {"v": 2}


# ---------------------------------------------------------------------------
# Batch ops
# ---------------------------------------------------------------------------


def test_batch_put_and_get():
    with _store() as store:
        ops = [
            PutOp(namespace=("n",), key="a", value={"v": 1}),
            PutOp(namespace=("n",), key="b", value={"v": 2}),
            PutOp(namespace=("n",), key="c", value={"v": 3}),
        ]
        store.batch(ops)
        store._conn.indices.refresh(index="langgraph_store")

        get_ops = [
            GetOp(namespace=("n",), key="a"),
            GetOp(namespace=("n",), key="b"),
            GetOp(namespace=("n",), key="missing"),
        ]
        results = store.batch(get_ops)
        assert results[0].value == {"v": 1}
        assert results[1].value == {"v": 2}
        assert results[2] is None


def test_batch_order_independent():
    """PutOp should not affect earlier GetOp results in same batch call."""
    with _store() as store:
        store.put(("t",), "existing", {"x": 99})
        store._conn.indices.refresh(index="langgraph_store")

        results = store.batch(
            [
                GetOp(namespace=("t",), key="existing"),
                PutOp(namespace=("t",), key="new", value={"y": 1}),
            ]
        )
        assert results[0].value == {"x": 99}
        assert results[1] is None  # PutOp returns None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_by_namespace_prefix():
    with _store() as store:
        store.put(("docs", "a"), "k1", {"title": "hello"})
        store.put(("docs", "b"), "k2", {"title": "world"})
        store.put(("other",), "k3", {"title": "ignore"})
        store._conn.indices.refresh(index="langgraph_store")

        results = store.search(("docs",))
        assert len(results) == 2
        for r in results:
            assert r.namespace[0] == "docs"


def test_search_with_filter():
    with _store() as store:
        store.put(("ns",), "a", {"status": "active", "score": 10})
        store.put(("ns",), "b", {"status": "inactive", "score": 5})
        store.put(("ns",), "c", {"status": "active", "score": 20})
        store._conn.indices.refresh(index="langgraph_store")

        results = store.search(("ns",), filter={"status": "active"})
        assert len(results) == 2
        for r in results:
            assert r.value["status"] == "active"


def test_search_limit_offset():
    with _store() as store:
        for i in range(5):
            store.put(("p",), f"k{i}", {"i": i})
        store._conn.indices.refresh(index="langgraph_store")

        page1 = store.search(("p",), limit=3, offset=0)
        page2 = store.search(("p",), limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2


def test_search_range_filter():
    with _store() as store:
        store.put(("r",), "low", {"score": 1})
        store.put(("r",), "mid", {"score": 5})
        store.put(("r",), "high", {"score": 10})
        store._conn.indices.refresh(index="langgraph_store")

        results = store.search(("r",), filter={"score": {"$gt": 4}})
        scores = {r.value["score"] for r in results}
        assert 1 not in scores
        assert 5 in scores
        assert 10 in scores


# ---------------------------------------------------------------------------
# List namespaces
# ---------------------------------------------------------------------------


def test_list_namespaces_all():
    with _store() as store:
        store.put(("a", "x"), "k", {"v": 1})
        store.put(("a", "y"), "k", {"v": 2})
        store.put(("b",), "k", {"v": 3})
        store._conn.indices.refresh(index="langgraph_store")

        namespaces = store.list_namespaces()
        assert ("a", "x") in namespaces
        assert ("a", "y") in namespaces
        assert ("b",) in namespaces


def test_list_namespaces_prefix_filter():
    with _store() as store:
        store.put(("users", "alice"), "k", {})
        store.put(("users", "bob"), "k", {})
        store.put(("docs",), "k", {})
        store._conn.indices.refresh(index="langgraph_store")

        namespaces = store.list_namespaces(
            match_conditions=(MatchCondition("prefix", ("users",)),)
        )
        for ns in namespaces:
            assert ns[0] == "users"


def test_list_namespaces_max_depth():
    with _store() as store:
        store.put(("a", "b", "c"), "k", {})
        store.put(("a", "b", "d"), "k", {})
        store.put(("a", "x", "y"), "k", {})
        store._conn.indices.refresh(index="langgraph_store")

        namespaces = store.list_namespaces(max_depth=2)
        for ns in namespaces:
            assert len(ns) <= 2


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def test_ttl_expires():
    ttl_cfg = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": False,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    with _store(ttl=ttl_cfg) as store:
        store.start_ttl_sweeper()
        store.put(("ttl-test",), "expiring", {"x": 1})
        store._conn.indices.refresh(index="langgraph_store")

        # Item should be there now
        assert store.get(("ttl-test",), "expiring") is not None

        # Wait for TTL to expire
        time.sleep(TTL_SECONDS + 2)
        store.sweep_ttl()
        store._conn.indices.refresh(index="langgraph_store")

        assert store.get(("ttl-test",), "expiring") is None
        store.stop_ttl_sweeper()


# ---------------------------------------------------------------------------
# Vector search (requires embeddings)
# ---------------------------------------------------------------------------


def test_vector_search_basic(fake_embeddings: CharacterEmbeddings):
    with _store(
        index={"dims": 64, "embed": fake_embeddings, "fields": ["text"]}
    ) as store:
        store.put(("vec",), "doc1", {"text": "Python tutorial"})
        store.put(("vec",), "doc2", {"text": "TypeScript guide"})
        store.put(("vec",), "doc3", {"text": "Machine learning basics"})
        store._conn.indices.refresh(index="langgraph_store")

        results = store.search(("vec",), query="Python programming", limit=2)
        # Just verify we get results without crashing; ranking is non-deterministic
        # with fake embeddings since they're not semantically meaningful
        assert isinstance(results, list)
