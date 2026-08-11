"""L5 offline option: deterministic hashing embeddings. No model, no network.

This is not a good embedding model and is not pretending to be one. It is a
hashed bag-of-words projection (the "hashing trick") with sublinear term
frequency and L2 normalisation. Lexical overlap between a query and a chunk
produces a real cosine signal, so retrieval genuinely works on a small corpus --
it just has no semantic understanding whatsoever.

Why it earns a place in the repo: the entire test suite runs against it, so
`pytest` needs no Ollama, no container, and no network, and it returns byte-
identical vectors on every machine. Any test that depends on a neural embedding
is a flaky test.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Sequence

from ...interfaces import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")

# Words carrying no retrieval signal; dropping them sharpens the cosine.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class HashEmbeddings(Embeddings):
    name = "hash"

    def __init__(self, *, dim: int = 512, **_: Any) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        """Stable bucket index plus a sign, so unrelated tokens can cancel."""
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            index, sign = self._bucket(token)
            # Sublinear tf: a word repeated 10x is not 10x more important.
            vec[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Empty/stopword-only text: a fixed unit vector beats a zero vector,
            # which would make cosine undefined downstream.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def available(self) -> bool:
        return True
