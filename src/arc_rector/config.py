"""Loads config.yaml, applies environment overrides, and builds the stack.

Precedence, lowest to highest:
    1. DEFAULTS below (so the package works with no config.yaml at all)
    2. config.yaml (or $ARC_CONFIG)
    3. environment variables: ARC_<LEVEL>=<adapter>  e.g. ARC_L4_VECTORSTORE=chroma
       and ARC_<LEVEL>__<SETTING>=<value>          e.g. ARC_L0_INFERENCE__MODEL=qwen2.5:3b

The env layer is what makes `ARC_L4_VECTORSTORE=chroma python -m arc_rector.demo`
work without editing a file -- used heavily by the swap demo and the Makefile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry
from .interfaces import (
    AgentDeps,
    AgentFramework,
    Embeddings,
    Evaluator,
    Guardrails,
    Inference,
    Loader,
    Memory,
    Reranker,
    Tracer,
    VectorStore,
)

DEFAULTS: dict[str, Any] = {
    "l0_inference": {"use": "ollama", "settings": {"model": "llama3.1:8b"}},
    "l1_observability": {"use": "langfuse", "settings": {}},
    "l1_eval": {"use": "ragas", "settings": {}},
    "l3_framework": {"use": "langgraph", "settings": {}},
    "l4_vectorstore": {"use": "qdrant", "settings": {}},
    "l5_embeddings": {"use": "nomic", "settings": {}},
    "l5_reranker": {"use": "none", "settings": {}},
    "l6_ingestion": {"use": "docling", "settings": {}},
    "l7_memory": {"use": "mem0", "settings": {}},
    "l8_guardrails": {"use": "guardrails-ai", "settings": {}},
    "pipeline": {
        "chunk_size": 900,
        "chunk_overlap": 150,
        "top_k": 4,
        "fetch_k": 12,
        "max_context_chars": 6000,
        "user_id": "demo-user",
        "corpus_dir": "corpus",
    },
}


def project_root() -> Path:
    """Repo root: the directory containing config.yaml, walking up from here."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


def _coerce(value: str) -> Any:
    """Turn an env string into bool/int/float where it plainly is one."""
    low = value.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_dotenv(root: Path) -> None:
    """Minimal .env reader. Avoids a dependency and never overwrites real env."""
    path = root / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=project_root)

    # -- accessors ---------------------------------------------------------
    def use(self, level: str) -> str:
        return str(self.raw.get(level, {}).get("use", DEFAULTS[level]["use"]))

    def settings(self, level: str, adapter: str | None = None) -> dict[str, Any]:
        """Settings for the active adapter of `level`.

        A level's `settings:` block holds values common to every adapter at that
        level (`collection`, `top_k`). Anything adapter-specific goes in a block
        named after the adapter and is merged on top:

            l4_vectorstore:
              use: qdrant
              settings:        {collection: arc_rector}
              qdrant:          {url: "http://localhost:6333"}
              chroma:          {path: .arc_rector/chroma}

        Without this split, swapping L4 to Chroma would hand it Qdrant's `url`
        and it would try to open an HTTP client against the Qdrant port. Every
        adapter tolerates unknown kwargs, so that failure is silent until it is
        very confusing -- which is exactly why the settings are scoped.
        """
        node = self.raw.get(level, {}) or {}
        common = dict(node.get("settings") or {})
        name = adapter or self.use(level)
        specific = node.get(name)
        if isinstance(specific, dict):
            return _deep_merge(common, specific)
        return common

    @property
    def pipeline(self) -> dict[str, Any]:
        return _deep_merge(DEFAULTS["pipeline"], self.raw.get("pipeline") or {})

    def selection(self) -> dict[str, str]:
        """{level: adapter} for every level -- what the banner prints."""
        return {lvl: self.use(lvl) for lvl in registry.LEVELS}

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        root = project_root()
        _load_dotenv(root)

        cfg_path = Path(path or os.environ.get("ARC_CONFIG") or (root / "config.yaml"))
        data: dict[str, Any] = {}
        if cfg_path.exists():
            import yaml

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        merged = _deep_merge(DEFAULTS, data)
        merged = cls._apply_env(merged)
        return cls(raw=merged, root=root)

    @staticmethod
    def _apply_env(data: dict[str, Any]) -> dict[str, Any]:
        known_levels = set(registry.LEVELS) | {"pipeline"}
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("ARC_"):
                continue
            body = env_key[4:]
            if "__" in body:
                lvl_part, _, setting = body.partition("__")
                level = lvl_part.lower()
                if level not in known_levels:
                    continue
                if level == "pipeline":
                    data.setdefault("pipeline", {})[setting.lower()] = _coerce(env_val)
                else:
                    # Env overrides land in the common block so they apply to
                    # whichever adapter is active, which is what someone typing
                    # ARC_L0_INFERENCE__MODEL on the command line means.
                    node = data.setdefault(level, {})
                    node.setdefault("settings", {})[setting.lower()] = _coerce(env_val)
            else:
                level = body.lower()
                if level in registry.LEVELS:
                    data.setdefault(level, {})["use"] = env_val
        return data

    # -- building ----------------------------------------------------------
    def _build(self, level: str, **extra: Any) -> Any:
        settings = _deep_merge(self.settings(level), extra)
        return registry.build(level, self.use(level), settings)

    def inference(self) -> Inference:
        return self._build(registry.L0_INFERENCE)

    def tracer(self) -> Tracer:
        return self._build(registry.L1_OBSERVABILITY)

    def evaluator(self) -> Evaluator:
        return self._build(registry.L1_EVAL)

    def framework(self) -> AgentFramework:
        return self._build(registry.L3_FRAMEWORK)

    def embeddings(self) -> Embeddings:
        return self._build(registry.L5_EMBEDDINGS)

    def reranker(self) -> Reranker:
        return self._build(registry.L5_RERANKER)

    def vectorstore(self, dim: int | None = None) -> VectorStore:
        extra = {"dim": dim} if dim is not None else {}
        return self._build(registry.L4_VECTORSTORE, **extra)

    def loader(self) -> Loader:
        return self._build(registry.L6_INGESTION)

    def memory(self) -> Memory:
        return self._build(registry.L7_MEMORY)

    def guardrails(self) -> Guardrails:
        return self._build(registry.L8_GUARDRAILS)

    def corpus_dir(self) -> Path:
        raw = str(self.pipeline.get("corpus_dir", "corpus"))
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def deps(self, *, embeddings: Embeddings | None = None, store: VectorStore | None = None) -> AgentDeps:
        """Construct every level once and bundle it for the agent framework."""
        emb = embeddings or self.embeddings()
        vec = store or self.vectorstore(dim=emb.dim)
        pipe = self.pipeline
        return AgentDeps(
            embeddings=emb,
            store=vec,
            inference=self.inference(),
            tracer=self.tracer(),
            memory=self.memory(),
            guardrails=self.guardrails(),
            reranker=self.reranker(),
            top_k=int(pipe["top_k"]),
            fetch_k=int(pipe["fetch_k"]),
            user_id=str(pipe["user_id"]),
            max_context_chars=int(pipe["max_context_chars"]),
        )


def load_config(path: str | Path | None = None) -> Config:
    return Config.load(path)
