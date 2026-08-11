"""L5 alternative: Jina Embeddings v2, open weights running locally.

This is *not* the Jina cloud API. Arc Rector is a zero-key project, so the
adapter loads Jina's Apache-2.0 open-weights checkpoint from the Hugging Face
Hub and runs it on your own machine through sentence-transformers.

Why you would pick it over the MiniLM default: `jina-embeddings-v2-small-en` is
33M parameters and 512 dimensions, but it uses ALiBi position encoding to
extrapolate to an 8192-token context. MiniLM truncates hard at 256 tokens, so a
900-character chunk is fine but a whole page is not. If your chunker produces
long chunks, Jina keeps the tail of the chunk; MiniLM silently discards it.
The cost is a torch install and the caveat below.

SUPPLY-CHAIN CAUTION: Jina v2 is a custom `JinaBERT` architecture, not stock
BERT, so it ships its modelling code on the Hub and requires
`trust_remote_code=True`. That flag downloads and executes third-party Python at
load time. Without it transformers instantiates a plain BERT and the load fails.
If that matters where you work, pin `revision` to a commit hash you have read,
or use the `sentence-transformers` adapter instead, which needs no remote code.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Sequence

from ...interfaces import Embeddings
from ...registry import require


def _discover_dim(model: Any) -> int:
    """Ask the loaded model for its width; never hardcode it."""
    # v5 renamed get_sentence_embedding_dimension() to get_embedding_dimension().
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, attr, None)
        if callable(method):
            value = method()
            if value:
                return int(value)
    # Last resort: measure one real vector. Always right, costs one forward pass.
    return len(model.encode(["dimension probe"], convert_to_numpy=True)[0])


class JinaEmbeddings(Embeddings):
    name = "jina"

    def __init__(
        self,
        *,
        model: str = "jinaai/jina-embeddings-v2-small-en",
        device: str | None = None,
        batch_size: int = 32,
        trust_remote_code: bool = True,
        revision: str | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self._encoder: Any = None
        self._dim = 0

    @property
    def _model(self) -> Any:
        """Lazy load: constructing the adapter must not pull torch or fetch weights."""
        if self._encoder is None:
            st = require("sentence-transformers", self.name)
            self._encoder = st.SentenceTransformer(
                self.model,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            self._dim = _discover_dim(self._encoder)
        return self._encoder

    @property
    def dim(self) -> int:
        """Reported by the checkpoint itself, so swapping v2-small for v2-base just works."""
        if self._dim == 0:
            _ = self._model
        return self._dim

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        # First argument stays positional: it is `sentences` on ST v2-v4, `inputs` on v5.
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in vec] for vec in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # Jina v2 is symmetric: no query/document prefix, unlike Nomic.
        return self._embed([text])[0]

    def available(self) -> bool:
        """Dependency check only -- weights download on first embed, not here."""
        return importlib.util.find_spec("sentence_transformers") is not None
