# langgraph-checkpoint-elasticsearch

Elasticsearch implementations of the LangGraph checkpoint saver and key-value store.

This is a **private** package — not published to PyPI. Install from source:

```bash
pip install -e libs/checkpoint-elasticsearch
```

## Components

### `ElasticsearchSaver` / `AsyncElasticsearchSaver`

Drop-in checkpointer that persists LangGraph thread state in three Elasticsearch indices:

| Index | Purpose |
|---|---|
| `langgraph_checkpoints` | One document per checkpoint |
| `langgraph_checkpoint_blobs` | Non-primitive channel values |
| `langgraph_checkpoint_writes` | Pending writes per task |

```python
from langgraph.checkpoint.elasticsearch import ElasticsearchSaver

with ElasticsearchSaver.from_conn_string("http://localhost:9200") as saver:
    saver.setup()
    graph = compiled_graph.compile(checkpointer=saver)
    result = graph.invoke(inputs, {"configurable": {"thread_id": "t1"}})
```

Async version:

```python
from langgraph.checkpoint.elasticsearch.aio import AsyncElasticsearchSaver

async with AsyncElasticsearchSaver.from_conn_string("http://localhost:9200") as saver:
    await saver.setup()
    graph = compiled_graph.compile(checkpointer=saver)
    result = await graph.ainvoke(inputs, {"configurable": {"thread_id": "t1"}})
```

### `ElasticsearchStore` / `AsyncElasticsearchStore`

Key-value store backed by Elasticsearch with optional vector search.

```python
from langgraph.store.elasticsearch import ElasticsearchStore

with ElasticsearchStore.from_conn_string("http://localhost:9200") as store:
    store.setup()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
```

With semantic (vector) search:

```python
from langchain.embeddings import init_embeddings
from langgraph.store.elasticsearch import ElasticsearchStore

with ElasticsearchStore.from_conn_string(
    "http://localhost:9200",
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],
    },
) as store:
    store.setup()
    store.put(("docs",), "doc1", {"text": "Python tutorial"})
    results = store.search(("docs",), query="programming guides")
```

## Running tests

```bash
cd libs/checkpoint-elasticsearch
make test
```

This starts a single-node Elasticsearch 8 instance on port 9201 via Docker Compose, runs the full test suite, and tears down the container.

## Dependencies

- `elasticsearch>=8.0.0` — official Python client
- `langgraph-checkpoint>=4.1.0` — base checkpoint interface
- `langgraph>=0.3.0` — store base interface
- `orjson>=3.11.5` — fast JSON serialisation
