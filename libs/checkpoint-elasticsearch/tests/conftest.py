"""Shared test fixtures for the Elasticsearch checkpoint and store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from elasticsearch import AsyncElasticsearch, Elasticsearch

DEFAULT_ES_URL = "http://localhost:9201"


# ---------------------------------------------------------------------------
# Sync ES client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def es_client() -> Iterator[Elasticsearch]:
    """Session-scoped sync Elasticsearch client."""
    client = Elasticsearch(DEFAULT_ES_URL)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Async ES client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="function")
async def async_es_client() -> AsyncIterator[AsyncElasticsearch]:
    """Function-scoped async Elasticsearch client."""
    client = AsyncElasticsearch(DEFAULT_ES_URL)
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# Minimal fake embeddings for vector search tests
# ---------------------------------------------------------------------------


class CharacterEmbeddings:
    """Deterministic fake embeddings based on character codes (for testing only)."""

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for i, ch in enumerate(text):
            vec[i % self.dims] += ord(ch) / 1000.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=64)
