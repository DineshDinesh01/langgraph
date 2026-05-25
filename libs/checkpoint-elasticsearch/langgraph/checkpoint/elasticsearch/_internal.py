"""Shared sync utility for the Elasticsearch checkpoint & store classes."""

from elasticsearch import Elasticsearch

# Type alias — accepts a plain Elasticsearch client.
# Unlike Postgres which has a connection pool type, the ES client
# handles connection pooling internally, so a single type covers both cases.
Conn = Elasticsearch
