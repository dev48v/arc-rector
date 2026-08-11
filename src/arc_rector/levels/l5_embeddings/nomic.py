"""L5 default: Nomic Embed via Ollama (Apache-2.0 weights).

`nomic-embed-text` is a 137M-parameter model with a 8192-token context that beats
`text-embedding-ada-002` on MTEB while being a 274 MB download you own outright.
Served through Ollama it needs no Python ML stack at all -- just HTTP -- which is
why it is the default here rather than a sentence-transformers model that would
drag in torch.

The one thing people get wrong: Nomic v1/v1.5 are *instruction-prefixed* models.
Corpus text must be embedded as `search_document: ...` and queries as
`search_query: ...`. Skip the prefixes and retrieval quality drops noticeably
while everything still appears to work, which makes it a nasty silent bug.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...interfaces import Embeddings
from ...registry import require

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicEmbeddings(Embeddings):
    name = "nomic"

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
        timeout: int = 120,
        use_prefixes: bool = True,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.timeout = timeout
        self.use_prefixes = use_prefixes
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", "nomic")
        return self._requests

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One call to Ollama's /api/embed, which accepts a batch."""
        response = self._http.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embeddings failed ({response.status_code}) at {self.base_url}: "
                f"{response.text[:300]}\n"
                f"Is Ollama running, and have you run `ollama pull {self.model}`?"
            )
        data = response.json()
        vectors = data.get("embeddings")
        if not vectors:
            raise RuntimeError(f"Ollama returned no embeddings: {str(data)[:300]}")
        self.dim = len(vectors[0])
        return [[float(x) for x in vec] for vec in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [DOCUMENT_PREFIX + t if self.use_prefixes else t for t in texts]
        return self._embed(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = QUERY_PREFIX + text if self.use_prefixes else text
        return self._embed([prepared])[0]

    def available(self) -> bool:
        try:
            response = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            names = {m.get("name", "").split(":")[0] for m in response.json().get("models", [])}
            return self.model.split(":")[0] in names
        except Exception:
            return False
