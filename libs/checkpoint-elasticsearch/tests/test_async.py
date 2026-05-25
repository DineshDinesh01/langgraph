# type: ignore
"""Async tests for AsyncElasticsearchSaver."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
from elasticsearch import AsyncElasticsearch
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.elasticsearch.aio import AsyncElasticsearchSaver
from tests.conftest import DEFAULT_ES_URL

pytestmark = pytest.mark.asyncio


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@asynccontextmanager
async def _saver():
    """Create a fresh AsyncElasticsearchSaver, wipe all docs before use."""
    async with AsyncElasticsearchSaver.from_conn_string(DEFAULT_ES_URL) as saver:
        await saver.setup()
        conn: AsyncElasticsearch = saver.conn
        await conn.delete_by_query(
            index=[
                "langgraph_checkpoints",
                "langgraph_checkpoint_blobs",
                "langgraph_checkpoint_writes",
            ],
            body={"query": {"match_all": {}}},
            refresh=True,
            ignore_unavailable=True,
        )
        yield saver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_data():
    config_1: RunnableConfig = {
        "configurable": {"thread_id": "thread-1", "checkpoint_id": "1", "checkpoint_ns": ""}
    }
    config_2: RunnableConfig = {
        "configurable": {"thread_id": "thread-2", "checkpoint_id": "2", "checkpoint_ns": ""}
    }
    config_3: RunnableConfig = {
        "configurable": {"thread_id": "thread-2", "checkpoint_id": "2-inner", "checkpoint_ns": "inner"}
    }
    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()
    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [
            {"source": "input", "step": 2, "score": 1},
            {"source": "loop", "step": 1, "score": None},
            {},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_setup_idempotent():
    async with _saver() as saver:
        await saver.setup()  # second call should not raise


@pytest.mark.asyncio
async def test_async_put_and_aget_tuple(test_data):
    async with _saver() as saver:
        cfg = test_data["configs"][0]
        chkpnt = test_data["checkpoints"][0]
        meta = test_data["metadata"][0]

        saved_cfg = await saver.aput(cfg, chkpnt, meta, {})
        await saver.conn.indices.refresh(index="langgraph_checkpoints")
        result = await saver.aget_tuple(saved_cfg)
        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]
        assert result.metadata["source"] == meta["source"]


@pytest.mark.asyncio
async def test_async_aget_tuple_missing():
    async with _saver() as saver:
        cfg: RunnableConfig = {
            "configurable": {"thread_id": "no-such-thread", "checkpoint_ns": ""}
        }
        result = await saver.aget_tuple(cfg)
        assert result is None


@pytest.mark.asyncio
async def test_async_alist_filter(test_data):
    async with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        await saver.aput(configs[0], checkpoints[0], metadata[0], {})
        await saver.aput(configs[1], checkpoints[1], metadata[1], {})
        await saver.aput(configs[2], checkpoints[2], metadata[2], {})
        await saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = [r async for r in saver.alist(None, filter={"source": "loop"})]
        assert len(results) == 1
        assert results[0].metadata["source"] == "loop"


@pytest.mark.asyncio
async def test_async_put_writes_and_get(test_data):
    async with _saver() as saver:
        cfg = test_data["configs"][0]
        saved_cfg = await saver.aput(cfg, test_data["checkpoints"][0], test_data["metadata"][0], {})
        await saver.aput_writes(
            saved_cfg,
            [("ch1", "v1"), ("ch2", "v2")],
            task_id="task-async-1",
        )
        await saver.conn.indices.refresh(
            index=["langgraph_checkpoints", "langgraph_checkpoint_writes"]
        )
        result = await saver.aget_tuple(saved_cfg)
        assert result is not None
        channels = {w[1] for w in result.pending_writes}
        assert "ch1" in channels
        assert "ch2" in channels


@pytest.mark.asyncio
async def test_async_delete_thread(test_data):
    async with _saver() as saver:
        cfg = test_data["configs"][0]
        await saver.aput(cfg, test_data["checkpoints"][0], test_data["metadata"][0], {})
        await saver.conn.indices.refresh(index="langgraph_checkpoints")

        await saver.adelete_thread("thread-1")
        await saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = [r async for r in saver.alist({"configurable": {"thread_id": "thread-1"}})]
        assert len(results) == 0


@pytest.mark.asyncio
async def test_async_list_by_thread(test_data):
    async with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        await saver.aput(configs[0], checkpoints[0], metadata[0], {})
        await saver.aput(configs[1], checkpoints[1], metadata[1], {})
        await saver.aput(configs[2], checkpoints[2], metadata[2], {})
        await saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = [r async for r in saver.alist({"configurable": {"thread_id": "thread-2"}})]
        assert len(results) == 2
        ns_set = {r.config["configurable"]["checkpoint_ns"] for r in results}
        assert ns_set == {"", "inner"}
