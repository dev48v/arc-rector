"""L4 -- vector database, Milvus implementation.

Same contract as every other L4 adapter, aimed at the case the others are not:
a cluster you expect to grow past one machine. The code below uses the modern
`MilvusClient` facade rather than the older ORM (`connections` + `Collection`),
which pymilvus now emits deprecation warnings for.

The gotcha, and the reason this adapter defaults to a server URI: Milvus Lite --
the file-backed mode you get by passing a local path such as "milvus.db" as the
uri -- does NOT run on Windows. The `milvus-lite` wheels are Linux and macOS
only, so on Windows you must point this adapter at a real server (Docker
`milvus-standalone` on http://localhost:19530, the default here).

Two smaller ones:

1. With `metric_type="COSINE"` Milvus reports similarity directly in each hit's
   `distance` field -- higher is already better, so unlike Chroma and pgvector
   there is no `1 - x` conversion. The field name is simply a misnomer.
2. The default consistency level is bounded-staleness, which makes a freshly
   upserted chunk briefly invisible to search and count. This adapter asks for
   "Strong" so an ingest-then-query demo behaves the way you expect.

Targets pymilvus 3.0.1.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...interfaces import VectorStore
from ...registry import require
from ...types import Chunk, Retrieved

_PIP_NAME = "pymilvus"
_ADAPTER = "milvus"


class MilvusStore(VectorStore):
    """Milvus-backed vector store built on the MilvusClient API."""

    name = "milvus"

    def __init__(
        self,
        *,
        uri: str = "http://localhost:19530",
        url: str = "",
        token: str = "",
        db_name: str = "",
        collection: str = "arc_rector",
        metric_type: str = "COSINE",
        consistency_level: str = "Strong",
        id_max_length: int = 64,
        **_: Any,
    ) -> None:
        self.uri = url or uri  # `url` is the generic config.yaml key
        self.token = token
        self.db_name = db_name
        self.collection = collection
        self.metric_type = metric_type.upper()
        self.consistency_level = consistency_level
        self.id_max_length = int(id_max_length)
        self._client_obj: Any = None

    # -- connection ---------------------------------------------------------

    @property
    def _client(self) -> Any:
        """Connect on first use, so merely selecting this adapter never fails."""
        if self._client_obj is None:
            pymilvus = require(_PIP_NAME, _ADAPTER, "pymilvus")
            self._client_obj = pymilvus.MilvusClient(
                uri=self.uri, token=self.token, db_name=self.db_name
            )
        return self._client_obj

    # -- VectorStore --------------------------------------------------------

    def ensure_collection(self, dim: int) -> None:
        client = self._client
        if client.has_collection(collection_name=self.collection):
            # A server restart leaves collections created but unloaded, and an
            # unloaded collection rejects both search and query.
            client.load_collection(collection_name=self.collection)
            return
        client.create_collection(
            collection_name=self.collection,
            dimension=int(dim),
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type=self.metric_type,
            auto_id=False,
            max_length=self.id_max_length,  # only read when the primary key is VARCHAR
            consistency_level=self.consistency_level,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        # `payload` is not a declared field: it lands in the dynamic field, which
        # quick-setup collections enable, and round-trips as JSON.
        rows = [
            {
                "id": chunk.chunk_id,
                "vector": [float(value) for value in vector],
                "payload": chunk.payload(),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        result = self._client.upsert(collection_name=self.collection, data=rows)
        if isinstance(result, dict):
            return int(result.get("upsert_count", len(rows)))
        return len(rows)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        results = self._client.search(
            collection_name=self.collection,
            data=[[float(value) for value in vector]],
            limit=max(1, int(top_k)),
            output_fields=["payload"],
        )
        hits = list(results[0]) if results else []
        retrieved: list[Retrieved] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            payload = entity.get("payload")
            payload = dict(payload) if isinstance(payload, dict) else {}
            payload["chunk_id"] = payload.get("chunk_id") or str(hit.get("id", ""))
            # COSINE metric: `distance` is already the similarity, higher is better.
            retrieved.append(
                Retrieved(chunk=Chunk.from_payload(payload), score=float(hit.get("distance", 0.0)))
            )
        return retrieved

    def count(self) -> int:
        client = self._client
        if not client.has_collection(collection_name=self.collection):
            return 0
        rows = client.query(collection_name=self.collection, output_fields=["count(*)"])
        if rows:
            return int(rows[0].get("count(*)", 0))
        return 0

    def drop(self) -> None:
        client = self._client
        if client.has_collection(collection_name=self.collection):
            client.drop_collection(collection_name=self.collection)

    def close(self) -> None:
        if self._client_obj is not None:
            self._client_obj.close()
            self._client_obj = None
