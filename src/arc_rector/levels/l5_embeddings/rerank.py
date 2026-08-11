"""L5 reranking: a cross-encoder second pass, and the no-op that is the default.

Retrieval is two-stage for a reason. A bi-encoder embeds the query and every
chunk *separately*, so a chunk's vector is fixed before your question exists --
cheap enough to index a million chunks, but it can only ever approximate
relevance. A cross-encoder puts the query and the chunk through the model
*together* and scores the pair, which is far more accurate and far too slow to
run over a whole corpus. So you fetch `fetch_k` cheaply, then rerank down to
`top_k` accurately. That is what `fetch_k: 12` and `top_k: 4` in config.yaml are
for: 12 vector hits go in, the 4 best come out.

The default is `none`, deliberately. Reranking costs a torch install, a model
download and roughly 50-200ms per query on CPU, and on a small demo corpus the
top 4 vector hits are usually already the right 4. Turn `cross-encoder` on when
recall is good but precision is not -- the answer is in the retrieved set, just
not in the slice you passed to the model.

Note that this file imports nothing third-party at module level, because
`NoReranker` is the default and must stay installable-free. sentence-transformers
is imported only inside `CrossEncoderReranker`, on first use.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Sequence

from ...interfaces import Reranker
from ...registry import require
from ...types import Retrieved


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(
        self,
        *,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._encoder: Any = None

    @property
    def _cross_encoder(self) -> Any:
        """Lazy load: building the stack must not pull torch when nothing reranks."""
        if self._encoder is None:
            st = require("sentence-transformers", self.name)
            kwargs: dict[str, Any] = {"device": self.device}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            self._encoder = st.CrossEncoder(self.model, **kwargs)
        return self._encoder

    def rerank(self, query: str, hits: Sequence[Retrieved], top_k: int) -> list[Retrieved]:
        ordered = list(hits)
        if not ordered or top_k <= 0:
            return []

        pairs = [[query, hit.text] for hit in ordered]
        # First argument stays positional: it is `sentences` on ST v2-v4, `inputs` on v5.
        scores = self._cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # The cross-encoder score replaces the vector score outright; the two are
        # different scales and must never be mixed or averaged.
        rescored = [
            Retrieved(chunk=hit.chunk, score=float(score))
            for hit, score in zip(ordered, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    def available(self) -> bool:
        """Dependency check only -- weights download on first rerank, not here."""
        return importlib.util.find_spec("sentence_transformers") is not None


class NoReranker(Reranker):
    """L5 default: pass the vector hits straight through, in the order given.

    Zero dependencies and zero latency. Keeping a real object here rather than a
    `None` check means the pipeline calls `reranker.rerank(...)` unconditionally,
    so switching reranking on is one word in config.yaml and no code path changes.
    """

    name = "none"

    def __init__(self, **_: Any) -> None:
        # Swallows settings meant for the cross-encoder so the same config block
        # works for either choice.
        pass

    def rerank(self, query: str, hits: Sequence[Retrieved], top_k: int) -> list[Retrieved]:
        if top_k <= 0:
            return []
        return list(hits)[:top_k]
