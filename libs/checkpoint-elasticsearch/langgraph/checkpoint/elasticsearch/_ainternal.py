"""Shared async utility for the Elasticsearch checkpoint & store classes."""

from elasticsearch import AsyncElasticsearch

# Type alias — accepts a plain AsyncElasticsearch client.
Conn = AsyncElasticsearch
