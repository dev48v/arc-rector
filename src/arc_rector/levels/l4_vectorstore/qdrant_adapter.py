"""L4 default: Qdrant (Apache-2.0).

Chosen as the default because it is a single ~275 MB container with no external
dependencies, it speaks cosine natively, and its payload model lets the whole
chunk ride along with the vector so retrieval needs exactly one round trip.

Gotcha worth knowing: Qdrant point IDs must be an unsigned integer or a UUID --
an arbitrary hex string is rejected. Arc Rector's chunk IDs are 32-character
SHA-1 prefixes, so they are reinterpreted as UUIDs on write (the original ID is
also kept in the payload, which is what the rest of the stack reads).
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from ...interfaces import VectorStore
from ...registry import require
from ...types import Chunk, Retrieved


def _point_id(chunk_id: str) -> str:
    """Map a 32-char hex chunk id onto a UUID, which Qdrant will accept."""
    try:
        return str(uuid.UUID(hex=chunk_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(
        self,
        *,
        collection: str = "arc_rector",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dim: int | None = None,
        timeout: int = 30,
        **_: Any,
    ) -> None:
        self.collection = collection
        self.url = url
        self.api_key = api_key
        self.dim = dim
        self.timeout = timeout
        self._c: Any = None
        self._models: Any = None

    @property
    def _client(self) -> Any:
        """Connect lazily so merely selecting this adapter never fails."""
        if self._c is None:
            qc = require("qdrant-client", "qdrant", "qdrant_client")
            self._models = require("qdrant-client", "qdrant", "qdrant_client.models")
            self._c = qc.QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
        return self._c

    def ensure_collection(self, dim: int) -> None:
        self.dim = dim
        client = self._client
        models = self._models
        if client.collection_exists(self.collection):
            info = client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing != dim:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' has dim {existing} but the active "
                    f"L5 embedding model produces dim {dim}. Swapping embedding models means "
                    f"re-ingesting: run `arc-rector ingest --reset`."
                )
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        models = self._models or require("qdrant-client", "qdrant", "qdrant_client.models")
        points = [
            models.PointStruct(id=_point_id(c.chunk_id), vector=list(v), payload=c.payload())
            for c, v in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        response = self._client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        out: list[Retrieved] = []
        for point in response.points:
            payload = point.payload or {}
            out.append(Retrieved(chunk=Chunk.from_payload(payload), score=float(point.score)))
        return out

    def count(self) -> int:
        if not self._client.collection_exists(self.collection):
            return 0
        return int(self._client.count(collection_name=self.collection, exact=True).count)

    def drop(self) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)

    def close(self) -> None:
        if self._c is not None:
            try:
                self._c.close()
            except Exception:
                pass
            self._c = None
