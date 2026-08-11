"""L5 alternative: any sentence-transformers checkpoint, in-process.

The generic local option. Whatever model id you put in `config.yaml` gets pulled
from the Hugging Face Hub and run inside this Python process -- no server, no
network at query time, no key. That makes it the reproducible choice for CI and
for laptops that will not be running Ollama.

The default `all-MiniLM-L6-v2` is the sensible baseline of the whole field:
22M parameters, 384 dimensions, Apache-2.0, ~90 MB. It is small enough that
embedding a corpus on CPU is a coffee-break rather than an afternoon.

The tradeoff against the Nomic default is dependencies versus context. This
adapter drags in torch and transformers (a multi-GB install, and the wheel you
get depends on your CUDA version), and MiniLM truncates input at 256 word
pieces, so anything past roughly a thousand characters of a chunk is dropped
before it is ever embedded. Nomic over HTTP avoids both problems but needs
Ollama running. Pick by which dependency you would rather own.
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


class SentenceTransformerEmbeddings(Embeddings):
    name = "sentence-transformers"

    def __init__(
        self,
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        # Off by default here: stock ST checkpoints never need to execute Hub code.
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
        """384 for MiniLM-L6, 768 for mpnet -- read off the checkpoint, not assumed."""
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
        # MiniLM is symmetric, so queries and documents share one encoding path.
        return self._embed([text])[0]

    def available(self) -> bool:
        """Dependency check only -- weights download on first embed, not here."""
        return importlib.util.find_spec("sentence_transformers") is not None
