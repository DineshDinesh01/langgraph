# type: ignore
"""Async tests for AsyncElasticsearchStore."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from elasticsearch import AsyncElasticsearch
from langgraph.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp

from langgraph.store.elasticsearch.aio import AsyncElasticsearchStore
from tests.conftest import DEFAULT_ES_URL

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _store(**kwargs):
    """Fresh async store — wipes the index before yielding."""
    async with AsyncElasticsearchStore.from_conn_string(DEFAULT_ES_URL, **kwargs) as store:
        await store.setup()
        conn: AsyncElasticsearch = store._conn
        await conn.delete_by_query(
            index="langgraph_store",
            body={"query": {"match_all": {}}},
            refresh=True,
            ignore_unavailable=True,
        )
        yield store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_put_and_get():
    async with _store() as store:
        await store.aput(("users",), "k1", {"role": "admin"})
        await store._conn.indices.refresh(index="langgraph_store")
        item = await store.aget(("users",), "k1")
        assert item is not None
        assert item.value == {"role": "admin"}


@pytest.mark.asyncio
async def test_async_get_missing():
    async with _store() as store:
        item = await store.aget(("no",), "key")
        assert item is None


@pytest.mark.asyncio
async def test_async_delete():
    async with _store() as store:
        await store.aput(("del",), "k", {"x": 1})
        await store._conn.indices.refresh(index="langgraph_store")
        await store.aput(("del",), "k", None)
        await store._conn.indices.refresh(index="langgraph_store")
        assert await store.aget(("del",), "k") is None


@pytest.mark.asyncio
async def test_async_batch_put_get():
    async with _store() as store:
        await store.abatch(
            [
                PutOp(("b",), "a", {"v": 1}),
                PutOp(("b",), "b", {"v": 2}),
            ]
        )
        await store._conn.indices.refresh(index="langgraph_store")

        results = await store.abatch(
            [
                GetOp(("b",), "a"),
                GetOp(("b",), "b"),
                GetOp(("b",), "missing"),
            ]
        )
        assert results[0].value == {"v": 1}
        assert results[1].value == {"v": 2}
        assert results[2] is None


@pytest.mark.asyncio
async def test_async_search():
    async with _store() as store:
        await store.aput(("docs", "sub"), "k1", {"type": "report"})
        await store.aput(("docs", "sub"), "k2", {"type": "memo"})
        await store.aput(("other",), "k3", {"type": "report"})
        await store._conn.indices.refresh(index="langgraph_store")

        results = await store.asearch(("docs",))
        assert len(results) == 2

        results_filtered = await store.asearch((), filter={"type": "report"})
        assert len(results_filtered) == 2


@pytest.mark.asyncio
async def test_async_list_namespaces():
    async with _store() as store:
        await store.aput(("x", "y"), "k", {})
        await store.aput(("x", "z"), "k", {})
        await store.aput(("w",), "k", {})
        await store._conn.indices.refresh(index="langgraph_store")

        namespaces = await store.alist_namespaces()
        ns_set = set(namespaces)
        assert ("x", "y") in ns_set
        assert ("x", "z") in ns_set
        assert ("w",) in ns_set


@pytest.mark.asyncio
async def test_async_list_namespaces_prefix():
    async with _store() as store:
        await store.aput(("cat", "a"), "k", {})
        await store.aput(("cat", "b"), "k", {})
        await store.aput(("dog",), "k", {})
        await store._conn.indices.refresh(index="langgraph_store")

        namespaces = await store.alist_namespaces(
            match_conditions=(MatchCondition("prefix", ("cat",)),)
        )
        for ns in namespaces:
            assert ns[0] == "cat"
