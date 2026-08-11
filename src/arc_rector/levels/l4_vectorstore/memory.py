"""L4 offline option: a pure-Python in-memory store.

No container, no dependency, no network. It exists so the test suite can
exercise the full retrieve path deterministically, and so `make demo-offline`
can prove the wiring on a machine with nothing installed.

Brute-force cosine over a list is O(n) per query -- fine for a demo corpus,
useless past a few thousand chunks. That is the honest trade, and it is exactly
why the other three L4 adapters exist.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ...interfaces import VectorStore
from ...types import Chunk, Retrieved


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, guarding the zero-vector case."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryStore(VectorStore):
    name = "memory"

    def __init__(self, *, dim: int | None = None, **_: Any) -> None:
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}

    def ensure_collection(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim and self._vectors:
            raise ValueError(f"store holds dim {self.dim} vectors, got {dim}")
        self.dim = dim

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors):
            self._vectors[chunk.chunk_id] = list(vector)
            self._chunks[chunk.chunk_id] = chunk
        return len(chunks)

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        scored = [
            Retrieved(chunk=self._chunks[cid], score=cosine(vector, vec))
            for cid, vec in self._vectors.items()
        ]
        # Sort by score, then chunk_id, so ties are deterministic across runs.
        scored.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return scored[:top_k]

    def count(self) -> int:
        return len(self._vectors)

    def drop(self) -> None:
        self._vectors.clear()
        self._chunks.clear()

    def close(self) -> None:
        pass
