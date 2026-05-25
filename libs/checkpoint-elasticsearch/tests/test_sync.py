# type: ignore
"""Sync tests for ElasticsearchSaver."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.elasticsearch import ElasticsearchSaver
from tests.conftest import DEFAULT_ES_URL


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@contextmanager
def _saver():
    """Create a fresh ElasticsearchSaver for each test."""
    with ElasticsearchSaver.from_conn_string(DEFAULT_ES_URL) as saver:
        saver.setup()
        # Wipe existing docs so tests are isolated
        conn: Elasticsearch = saver.conn
        conn.delete_by_query(
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
    metadata_1: CheckpointMetadata = {"source": "input", "step": 2, "score": 1}
    metadata_2: CheckpointMetadata = {"source": "loop", "step": 1, "score": None}
    metadata_3: CheckpointMetadata = {}
    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_setup_is_idempotent():
    """Calling setup() twice must not raise."""
    with _saver() as saver:
        saver.setup()


def test_put_and_get_tuple(test_data):
    with _saver() as saver:
        cfg, chkpnt, meta = (
            test_data["configs"][0],
            test_data["checkpoints"][0],
            test_data["metadata"][0],
        )
        saved_cfg = saver.put(cfg, chkpnt, meta, {})
        time.sleep(0.5)  # allow ES to refresh
        result = saver.get_tuple(saved_cfg)
        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]
        assert result.metadata["source"] == meta["source"]


def test_get_tuple_missing_returns_none():
    with _saver() as saver:
        cfg: RunnableConfig = {
            "configurable": {
                "thread_id": "no-such-thread",
                "checkpoint_ns": "",
            }
        }
        result = saver.get_tuple(cfg)
        assert result is None


def test_list_returns_all(test_data):
    with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        # ES index needs a moment to become searchable
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = list(saver.list(None, filter={}))
        assert len(results) == 3


def test_list_filter_by_source(test_data):
    with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = list(saver.list(None, filter={"source": "input"}))
        assert len(results) == 1
        assert results[0].metadata["source"] == "input"


def test_list_filter_by_thread(test_data):
    with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = list(
            saver.list({"configurable": {"thread_id": "thread-2"}})
        )
        assert len(results) == 2
        ns_set = {r.config["configurable"]["checkpoint_ns"] for r in results}
        assert ns_set == {"", "inner"}


def test_put_writes_and_get_tuple(test_data):
    with _saver() as saver:
        cfg = test_data["configs"][0]
        chkpnt = test_data["checkpoints"][0]
        meta = test_data["metadata"][0]

        saved_cfg = saver.put(cfg, chkpnt, meta, {})
        saver.put_writes(
            saved_cfg,
            [("channel_a", "val_a"), ("channel_b", "val_b")],
            task_id="task-1",
        )
        saver.conn.indices.refresh(
            index=["langgraph_checkpoints", "langgraph_checkpoint_writes"]
        )
        result = saver.get_tuple(saved_cfg)
        assert result is not None
        channels = {w[1] for w in result.pending_writes}
        assert "channel_a" in channels
        assert "channel_b" in channels


def test_delete_thread(test_data):
    with _saver() as saver:
        cfg = test_data["configs"][0]
        saved_cfg = saver.put(
            cfg, test_data["checkpoints"][0], test_data["metadata"][0], {}
        )
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        saver.delete_thread("thread-1")
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = list(saver.list({"configurable": {"thread_id": "thread-1"}}))
        assert len(results) == 0


def test_multiple_checkpoints_same_thread():
    """Listing a thread returns checkpoints in descending checkpoint_id order."""
    with _saver() as saver:
        cfg_base: RunnableConfig = {
            "configurable": {"thread_id": "multi-thread", "checkpoint_ns": ""}
        }
        c1 = empty_checkpoint()
        c2 = create_checkpoint(c1, {}, 1)
        c3 = create_checkpoint(c2, {}, 2)

        saver.put(cfg_base, c1, {"source": "input", "step": 0}, {})
        saver.put(cfg_base, c2, {"source": "loop", "step": 1}, {})
        saver.put(cfg_base, c3, {"source": "loop", "step": 2}, {})
        saver.conn.indices.refresh(index="langgraph_checkpoints")

        results = list(saver.list(cfg_base))
        assert len(results) == 3
        steps = [r.metadata["step"] for r in results]
        assert steps == sorted(steps, reverse=True)
