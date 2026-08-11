"""The adapter registry: how one line of config.yaml swaps a whole layer.

Every level of the stack (L0..L8) declares a base interface. Concrete adapters
register themselves against a short name. `resolve(level, name)` returns the
class, and `build(level, name, settings)` instantiates it.

Adapters are imported lazily: importing `arc_rector` must never require every
optional dependency to be installed. A missing dependency surfaces as a clear
`AdapterUnavailable` at build time, never as a silent fallback.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# level -> {adapter name -> "module.path:ClassName"}
_LAZY: dict[str, dict[str, str]] = {}
# level -> {adapter name -> class} (populated by @register or after lazy import)
_LOADED: dict[str, dict[str, type]] = {}


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter is known but its dependency is not installed."""

    def __init__(self, adapter: str, package: str, detail: str = "") -> None:
        msg = (
            f"Adapter '{adapter}' needs a package that is not installed: {package}\n"
            f"    Install it with:  pip install {package}"
        )
        if detail:
            msg += f"\n    Detail: {detail}"
        super().__init__(msg)
        self.adapter = adapter
        self.package = package


class UnknownAdapter(KeyError):
    """Raised when config names an adapter that does not exist for that level."""

    def __init__(self, level: str, name: str, known: list[str]) -> None:
        super().__init__(
            f"Unknown adapter '{name}' for level '{level}'. Known: {', '.join(sorted(known))}"
        )


def declare(level: str, name: str, target: str) -> None:
    """Declare an adapter lazily as 'module.path:ClassName' without importing it."""
    _LAZY.setdefault(level, {})[name] = target


def register(level: str, name: str) -> Callable[[type], type]:
    """Decorator form, for adapters defined in an already-imported module."""

    def _inner(cls: type) -> type:
        _LOADED.setdefault(level, {})[name] = cls
        return cls

    return _inner


def known(level: str) -> list[str]:
    return sorted(set(_LAZY.get(level, {})) | set(_LOADED.get(level, {})))


def resolve(level: str, name: str) -> type:
    """Return the adapter class for `name`, importing its module on first use."""
    loaded = _LOADED.get(level, {})
    if name in loaded:
        return loaded[name]

    target = _LAZY.get(level, {}).get(name)
    if target is None:
        raise UnknownAdapter(level, name, known(level))

    module_path, _, cls_name = target.partition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    _LOADED.setdefault(level, {})[name] = cls
    return cls


def build(level: str, name: str, settings: dict[str, Any] | None = None) -> Any:
    """Instantiate the adapter named `name` for `level` with keyword settings."""
    cls = resolve(level, name)
    return cls(**(settings or {}))


def require(package: str, adapter: str, import_name: str | None = None) -> Any:
    """Import `package` for `adapter`, converting ImportError into AdapterUnavailable.

    `import_name` is the module to import when it differs from the pip name
    (e.g. pip install mem0ai -> import mem0).
    """
    module_name = import_name or package.replace("-", "_")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise AdapterUnavailable(adapter, package, str(exc)) from exc


# --------------------------------------------------------------------------
# Level names. Using constants keeps config keys and registry keys in sync.
# --------------------------------------------------------------------------
L0_INFERENCE = "l0_inference"
L1_OBSERVABILITY = "l1_observability"
L1_EVAL = "l1_eval"
L3_FRAMEWORK = "l3_framework"
L4_VECTORSTORE = "l4_vectorstore"
L5_EMBEDDINGS = "l5_embeddings"
L5_RERANKER = "l5_reranker"
L6_INGESTION = "l6_ingestion"
L7_MEMORY = "l7_memory"
L8_GUARDRAILS = "l8_guardrails"

LEVELS = [
    L0_INFERENCE,
    L1_OBSERVABILITY,
    L1_EVAL,
    L3_FRAMEWORK,
    L4_VECTORSTORE,
    L5_EMBEDDINGS,
    L5_RERANKER,
    L6_INGESTION,
    L7_MEMORY,
    L8_GUARDRAILS,
]


def _declare_all() -> None:
    """Single place listing every adapter shipped with Arc Rector."""
    p = "arc_rector.levels"

    # L0 - inference & deployment
    declare(L0_INFERENCE, "ollama", f"{p}.l0_inference.ollama_adapter:OllamaInference")
    declare(L0_INFERENCE, "nim", f"{p}.l0_inference.nim:NimInference")
    declare(L0_INFERENCE, "vllm", f"{p}.l0_inference.vllm_adapter:VllmInference")
    declare(L0_INFERENCE, "llamacpp", f"{p}.l0_inference.llamacpp:LlamaCppInference")
    declare(L0_INFERENCE, "echo", f"{p}.l0_inference.echo:EchoInference")

    # L1 - observability
    declare(L1_OBSERVABILITY, "langfuse", f"{p}.l1_observability.langfuse_adapter:LangfuseTracer")
    declare(L1_OBSERVABILITY, "phoenix", f"{p}.l1_observability.phoenix:PhoenixTracer")
    declare(L1_OBSERVABILITY, "none", f"{p}.l1_observability.noop:NoopTracer")

    # L1 - evaluation
    declare(L1_EVAL, "ragas", f"{p}.l1_eval.ragas_adapter:RagasEvaluator")
    declare(L1_EVAL, "deepeval", f"{p}.l1_eval.deepeval_adapter:DeepEvalEvaluator")
    declare(L1_EVAL, "builtin", f"{p}.l1_eval.builtin:BuiltinEvaluator")

    # L3 - agent frameworks
    declare(L3_FRAMEWORK, "langgraph", f"{p}.l3_framework.langgraph_adapter:LangGraphAgent")
    declare(L3_FRAMEWORK, "llamaindex", f"{p}.l3_framework.llamaindex_adapter:LlamaIndexAgent")
    declare(L3_FRAMEWORK, "haystack", f"{p}.l3_framework.haystack_adapter:HaystackAgent")
    declare(L3_FRAMEWORK, "crewai", f"{p}.l3_framework.crewai_adapter:CrewAIAgent")
    declare(L3_FRAMEWORK, "dspy", f"{p}.l3_framework.dspy_adapter:DspyAgent")
    declare(L3_FRAMEWORK, "simple", f"{p}.l3_framework.simple:SimpleAgent")

    # L4 - vector stores
    declare(L4_VECTORSTORE, "qdrant", f"{p}.l4_vectorstore.qdrant_adapter:QdrantStore")
    declare(L4_VECTORSTORE, "chroma", f"{p}.l4_vectorstore.chroma_adapter:ChromaStore")
    declare(L4_VECTORSTORE, "pgvector", f"{p}.l4_vectorstore.pgvector_adapter:PgVectorStore")
    declare(L4_VECTORSTORE, "milvus", f"{p}.l4_vectorstore.milvus_adapter:MilvusStore")
    declare(L4_VECTORSTORE, "memory", f"{p}.l4_vectorstore.memory:InMemoryStore")

    # L5 - embeddings & reranking
    declare(L5_EMBEDDINGS, "nomic", f"{p}.l5_embeddings.nomic:NomicEmbeddings")
    declare(L5_EMBEDDINGS, "jina", f"{p}.l5_embeddings.jina:JinaEmbeddings")
    declare(
        L5_EMBEDDINGS,
        "sentence-transformers",
        f"{p}.l5_embeddings.sentence_transformers_adapter:SentenceTransformerEmbeddings",
    )
    declare(L5_EMBEDDINGS, "hash", f"{p}.l5_embeddings.hashing:HashEmbeddings")
    declare(L5_RERANKER, "cross-encoder", f"{p}.l5_embeddings.rerank:CrossEncoderReranker")
    declare(L5_RERANKER, "none", f"{p}.l5_embeddings.rerank:NoReranker")

    # L6 - ingestion & parsing
    declare(L6_INGESTION, "docling", f"{p}.l6_ingestion.docling_adapter:DoclingLoader")
    declare(L6_INGESTION, "unstructured", f"{p}.l6_ingestion.unstructured_adapter:UnstructuredLoader")
    declare(L6_INGESTION, "firecrawl", f"{p}.l6_ingestion.firecrawl_adapter:FirecrawlLoader")
    declare(L6_INGESTION, "scrapy", f"{p}.l6_ingestion.scrapy_adapter:ScrapyLoader")
    declare(L6_INGESTION, "plaintext", f"{p}.l6_ingestion.plaintext:PlaintextLoader")

    # L7 - memory & context
    declare(L7_MEMORY, "mem0", f"{p}.l7_memory.mem0_adapter:Mem0Memory")
    declare(L7_MEMORY, "zep", f"{p}.l7_memory.zep_adapter:ZepMemory")
    declare(L7_MEMORY, "letta", f"{p}.l7_memory.letta_adapter:LettaMemory")
    declare(L7_MEMORY, "cognee", f"{p}.l7_memory.cognee_adapter:CogneeMemory")
    declare(L7_MEMORY, "graphiti", f"{p}.l7_memory.graphiti_adapter:GraphitiMemory")
    declare(L7_MEMORY, "local", f"{p}.l7_memory.local:LocalMemory")
    declare(L7_MEMORY, "none", f"{p}.l7_memory.local:NoMemory")

    # L8 - safety & guardrails
    declare(L8_GUARDRAILS, "guardrails-ai", f"{p}.l8_guardrails.guardrails_adapter:GuardrailsAI")
    declare(L8_GUARDRAILS, "nemo", f"{p}.l8_guardrails.nemo_adapter:NemoGuardrails")
    declare(L8_GUARDRAILS, "llamafirewall", f"{p}.l8_guardrails.llamafirewall_adapter:LlamaFirewallGuard")
    declare(L8_GUARDRAILS, "llamaguard", f"{p}.l8_guardrails.llamaguard_adapter:LlamaGuard")
    declare(L8_GUARDRAILS, "builtin", f"{p}.l8_guardrails.builtin:BuiltinGuardrails")
    declare(L8_GUARDRAILS, "none", f"{p}.l8_guardrails.builtin:NoGuardrails")


_declare_all()
