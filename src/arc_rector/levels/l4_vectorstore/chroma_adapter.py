"""L4 -- vector database, Chroma implementation.

This level owns exactly one job: keep chunk vectors and answer "which chunks are
nearest to this query vector". Nothing above L4 knows Chroma exists; swapping
`l4_vectorstore.use` in config.yaml to `chroma` is the entire migration.

Two gotchas make Chroma different from the other stores here:

1. Chroma's default distance space is squared L2, not cosine, and the space is
   frozen at creation time -- you cannot change it on an existing collection.
   The space is set through `configuration={"hnsw": {"space": "cosine"}}` on
   current clients; releases before 1.0 used the flat `metadata={"hnsw:space":
   ...}` key instead, which is why the create call below tries both.
2. Chroma stores metadata values as scalars only (str/int/float/bool) and
   rejects `None`, so the chunk payload is filtered before it goes in.

Chroma returns cosine *distance*. Scores are converted with `1 - distance` so
that, exactly as in every other L4 adapter, higher means more similar.

Targets chromadb 1.5.9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from ...interfaces import VectorStore
from ...registry import require
from ...types import Chunk, Retrieved

_PIP_NAME = "chromadb"
_ADAPTER = "chroma"
_SCALARS = (str, int, float, bool)


class ChromaStore(VectorStore):
    """Chroma-backed vector store: embedded and persistent by default."""

    name = "chroma"

    def __init__(
        self,
        *,
        collection: str = "arc_rector",
        path: str = ".arc_rector/chroma",
        host: str = "",
        port: int = 8000,
        ssl: bool = False,
        url: str = "",
        distance: str = "cosine",
        **_: Any,
    ) -> None:
        self.collection = collection
        self.path = path
        self.host = host
        self.port = int(port)
        self.ssl = bool(ssl)
        self.distance = distance
        if url:
            self._apply_url(url)
        self._client_obj: Any = None
        self._coll: Any = None

    def _apply_url(self, url: str) -> None:
        """Accept `url: http://host:8000` from config.yaml as host/port/ssl."""
        parsed = urlparse(url if "//" in url else f"//{url}")
        self.host = parsed.hostname or self.host
        self.ssl = parsed.scheme == "https"
        self.port = parsed.port or (443 if self.ssl else self.port)

    # -- connection ---------------------------------------------------------

    @property
    def _client(self) -> Any:
        """Connect on first use, so merely selecting this adapter never fails."""
        if self._client_obj is None:
            chromadb = require(_PIP_NAME, _ADAPTER, "chromadb")
            if self.host:
                self._client_obj = chromadb.HttpClient(
                    host=self.host, port=self.port, ssl=self.ssl
                )
            else:
                Path(self.path).mkdir(parents=True, exist_ok=True)
                self._client_obj = chromadb.PersistentClient(path=self.path)
        return self._client_obj

    @property
    def _handle(self) -> Any:
        if self._coll is None:
            self._coll = self._open_collection()
        return self._coll

    def _open_collection(self) -> Any:
        client = self._client
        try:
            return client.get_or_create_collection(
                name=self.collection,
                configuration={"hnsw": {"space": self.distance}},
                embedding_function=None,
            )
        except TypeError:
            # chromadb < 1.0 has no `configuration=`; the space was a metadata key.
            return client.get_or_create_collection(
                name=self.collection,
                metadata={"hnsw:space": self.distance},
                embedding_function=None,
            )

    # -- VectorStore --------------------------------------------------------

    def ensure_collection(self, dim: int) -> None:
        # Chroma infers dimensionality from the first vector written, so `dim`
        # is not part of the create call; the collection is still made eagerly.
        self._coll = self._open_collection()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        self._handle.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[[float(x) for x in vector] for vector in vectors],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._as_metadata(chunk.payload()) for chunk in chunks],
        )
        return len(chunks)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        result = self._handle.query(
            query_embeddings=[[float(x) for x in vector]],
            n_results=max(1, int(top_k)),
            include=["documents", "metadatas", "distances"],
        )
        # Every key is a list-of-lists, one inner list per query vector.
        ids = self._first(result, "ids")
        documents = self._first(result, "documents")
        metadatas = self._first(result, "metadatas")
        distances = self._first(result, "distances")

        hits: list[Retrieved] = []
        for index, chunk_id in enumerate(ids):
            payload = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            payload["chunk_id"] = payload.get("chunk_id") or chunk_id
            payload["text"] = (documents[index] or "") if index < len(documents) else ""
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(Retrieved(chunk=Chunk.from_payload(payload), score=1.0 - distance))
        return hits

    def count(self) -> int:
        return int(self._handle.count())

    def drop(self) -> None:
        try:
            self._client.delete_collection(name=self.collection)
        except Exception:
            # Dropping something that was never created is not an error here.
            pass
        self._coll = None

    def close(self) -> None:
        # Chroma's embedded client has no explicit close; releasing the handles
        # is what lets a later `drop()` on the same path succeed on Windows.
        self._coll = None
        self._client_obj = None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _first(result: dict[str, Any], key: str) -> list[Any]:
        rows = result.get(key) or [[]]
        return list(rows[0] or []) if rows else []

    @staticmethod
    def _as_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata takes scalars only; `text` travels in `documents`."""
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "text" or value is None:
                continue
            clean[key] = value if isinstance(value, _SCALARS) else json.dumps(value, default=str)
        return clean
