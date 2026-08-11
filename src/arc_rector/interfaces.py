"""The nine interfaces. One per level of the stack.

Every adapter in `arc_rector.levels.*` subclasses exactly one of these. If you
want to add a new implementation of a level, subclass the interface, add one
line to `registry._declare_all()`, and name it in `config.yaml`. Nothing else in
the codebase needs to change -- that is the entire point of this project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .types import Chunk, Document, GuardResult, MemoryRecord, Retrieved


class Inference(ABC):
    """L0 -- inference & deployment. Turns a prompt into text."""

    name: str = "inference"

    @abstractmethod
    def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        """Return a completion for `prompt`."""

    def available(self) -> bool:
        """Cheap liveness probe. Adapters override with a real health check."""
        return True


class Tracer(ABC):
    """L1 -- observability. Records nested spans for every step of a run."""

    name: str = "tracer"

    @abstractmethod
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator["SpanHandle"]:
        """Open a span; yields a handle whose `.update(output=...)` records results."""

    def flush(self) -> None:
        """Force-send buffered telemetry. Important for short-lived processes."""

    def trace_url(self) -> str:
        """Human-visitable URL for the last trace, when the backend has a UI."""
        return ""

    def last_trace_id(self) -> str:
        return ""


class SpanHandle(ABC):
    """Handle yielded by `Tracer.span`."""

    @abstractmethod
    def update(self, **fields: Any) -> None:
        """Attach output/metadata to the open span."""


class Evaluator(ABC):
    """L1 -- evaluation. Scores a set of question/answer/context samples."""

    name: str = "evaluator"

    @abstractmethod
    def evaluate(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """`samples` carry user_input, response, retrieved_contexts, reference.

        Returns {"metrics": {name: score}, "per_sample": [...], "backend": str}.
        """


class AgentFramework(ABC):
    """L3 -- agent framework. Orchestrates retrieve -> reason -> answer."""

    name: str = "framework"

    @abstractmethod
    def run(self, question: str, deps: "AgentDeps") -> "Answer":
        """Execute one full turn and return an Answer with citations."""


class VectorStore(ABC):
    """L4 -- vector database. Stores chunk vectors and does similarity search."""

    name: str = "vectorstore"

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Create the collection/table/index if it does not exist."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or replace chunks. Returns the number written."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 5) -> list[Retrieved]:
        """Return the `top_k` nearest chunks, highest score first."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""

    def drop(self) -> None:
        """Delete the collection. Used by tests and `arc-rector ingest --reset`."""

    def close(self) -> None:
        """Release connections."""


class Embeddings(ABC):
    """L5 -- embeddings. Text to vectors."""

    name: str = "embeddings"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks (some models use a distinct document prefix)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models use a distinct query prefix)."""

    def available(self) -> bool:
        return True


class Reranker(ABC):
    """L5 -- reranking. Reorders retrieved chunks with a cross-encoder."""

    name: str = "reranker"

    @abstractmethod
    def rerank(self, query: str, hits: Sequence[Retrieved], top_k: int) -> list[Retrieved]:
        """Return up to `top_k` hits in improved relevance order."""


class Loader(ABC):
    """L6 -- ingestion & parsing. Files or URLs to Documents."""

    name: str = "loader"

    @abstractmethod
    def load(self, source: str) -> Document:
        """Parse one file path or URL into a Document."""

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Whether this loader can handle the given path/URL."""


class Memory(ABC):
    """L7 -- memory & context. Facts that persist across turns."""

    name: str = "memory"

    @abstractmethod
    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        """Store salient facts from a turn. Returns the stored fact texts."""

    @abstractmethod
    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        """Recall facts relevant to `query` for `user_id`."""

    def reset(self, user_id: str = "") -> None:
        """Forget everything (for a user, or globally when user_id is empty)."""


class Guardrails(ABC):
    """L8 -- safety & guardrails. Validates model input and output."""

    name: str = "guardrails"

    @abstractmethod
    def check_input(self, text: str) -> GuardResult:
        """Validate a user question before it reaches retrieval or the model."""

    @abstractmethod
    def check_output(self, text: str, context: str = "") -> GuardResult:
        """Validate a generated answer before it reaches the user."""

    def check_context(self, context: str) -> GuardResult:
        """Scan retrieved documents for embedded instructions (indirect injection).

        Default is permissive so existing adapters keep working; `allowed=False`
        means "instruction-shaped text was found", not "refuse to answer".
        """
        return GuardResult(allowed=True, text=context, validator=self.name)


class AgentDeps:
    """Everything an L3 framework adapter needs, already built from config.

    Passing one object keeps framework adapters short: they orchestrate, they do
    not construct. Swapping L4 or L5 therefore needs no change in any L3 adapter.
    """

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        store: VectorStore,
        inference: Inference,
        tracer: Tracer,
        memory: Memory,
        guardrails: Guardrails,
        reranker: Reranker,
        top_k: int = 4,
        fetch_k: int = 12,
        user_id: str = "demo-user",
        max_context_chars: int = 6000,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.inference = inference
        self.tracer = tracer
        self.memory = memory
        self.guardrails = guardrails
        self.reranker = reranker
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.user_id = user_id
        self.max_context_chars = max_context_chars


# Imported last to avoid a circular import at module load time.
from .types import Answer  # noqa: E402  (re-exported for adapter type hints)

__all__ = [
    "AgentDeps",
    "AgentFramework",
    "Answer",
    "Chunk",
    "Document",
    "Embeddings",
    "Evaluator",
    "GuardResult",
    "Guardrails",
    "Inference",
    "Loader",
    "Memory",
    "MemoryRecord",
    "Reranker",
    "Retrieved",
    "SpanHandle",
    "Tracer",
    "VectorStore",
]
