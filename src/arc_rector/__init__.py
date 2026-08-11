"""Arc Rector -- a complete agentic RAG stack built entirely from open source.

Nine levels, each a swappable adapter behind one interface, each with a default
that runs with no vendor account and no API key:

    L0 inference      Ollama            L5 embeddings   Nomic Embed
    L1 observability  Langfuse          L5 reranking    (off by default)
    L1 evaluation     Ragas             L6 ingestion    Docling
    L2 models         Llama 3.1 8B      L7 memory       Mem0
    L3 framework      LangGraph         L8 guardrails   Guardrails AI
    L4 vector store   Qdrant

Typical use:

    from arc_rector import load_config, ask
    print(ask("What is Arc Rector?").render())

Swap any level from config.yaml, or per-run from the environment:

    ARC_L4_VECTORSTORE=chroma python -m arc_rector.demo
"""

from __future__ import annotations

from .config import Config, load_config
from .types import Answer, Chunk, Citation, Document, GuardResult, MemoryRecord, Retrieved

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Config",
    "Document",
    "GuardResult",
    "MemoryRecord",
    "Retrieved",
    "__version__",
    "ask",
    "load_config",
]


def ask(
    question: str,
    config: Config | None = None,
    *,
    deps: object | None = None,
    framework: object | None = None,
    **deps_overrides: object,
) -> Answer:
    """Answer one question with the stack described by config.yaml.

    Convenience wrapper: builds every level, runs the active L3 framework, and
    flushes telemetry so a short-lived script still records its trace.

    `deps` and `framework` let a long-lived process (the web server) build the
    stack once and reuse it, without forking a second copy of this logic. One
    entry point for the CLI and the UI is what stops the two drifting apart.
    """
    cfg = config or load_config()
    built = deps if deps is not None else cfg.deps(**deps_overrides)  # type: ignore[arg-type]
    agent = framework if framework is not None else cfg.framework()
    try:
        return agent.run(question, built)  # type: ignore[union-attr]
    finally:
        built.tracer.flush()  # type: ignore[union-attr]
