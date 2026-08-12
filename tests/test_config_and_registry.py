"""The swap mechanism itself: config precedence and adapter resolution.

If these break, the project's central claim -- change one line, swap a layer --
stops being true, so they are worth testing as carefully as the RAG logic.
"""

from __future__ import annotations

import pytest

from arc_rector import registry
from arc_rector.config import Config, _coerce, _deep_merge
from arc_rector.interfaces import (
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

# Each level, its interface, and the adapters that need no third-party package.
OFFLINE_ADAPTERS = {
    registry.L0_INFERENCE: (Inference, ["echo"]),
    registry.L1_OBSERVABILITY: (Tracer, ["none"]),
    registry.L1_EVAL: (Evaluator, ["builtin"]),
    registry.L3_FRAMEWORK: (AgentFramework, ["simple"]),
    registry.L4_VECTORSTORE: (VectorStore, ["memory"]),
    registry.L5_EMBEDDINGS: (Embeddings, ["hash"]),
    registry.L5_RERANKER: (Reranker, ["none"]),
    registry.L6_INGESTION: (Loader, ["plaintext"]),
    registry.L7_MEMORY: (Memory, ["local", "none"]),
    registry.L8_GUARDRAILS: (Guardrails, ["builtin", "none"]),
}


def test_every_level_has_at_least_one_adapter():
    for level in registry.LEVELS:
        assert registry.known(level), f"{level} has no adapters registered"


@pytest.mark.parametrize(
    "level,name",
    [(lvl, n) for lvl, (_, names) in OFFLINE_ADAPTERS.items() for n in names],
)
def test_offline_adapters_resolve_and_implement_their_interface(level, name):
    interface = OFFLINE_ADAPTERS[level][0]
    cls = registry.resolve(level, name)
    assert issubclass(cls, interface)
    # Instantiating proves no abstract method was left unimplemented.
    assert isinstance(cls(), interface)


def test_unknown_adapter_raises_with_the_known_list():
    with pytest.raises(registry.UnknownAdapter) as exc:
        registry.resolve(registry.L4_VECTORSTORE, "not-a-real-store")
    assert "qdrant" in str(exc.value)


def test_adapter_unavailable_names_the_pip_package():
    error = registry.AdapterUnavailable("thing", "some-package")
    assert "pip install some-package" in str(error)


def test_declared_adapters_all_have_an_import_target():
    for level in registry.LEVELS:
        for name in registry.known(level):
            assert registry._LAZY.get(level, {}).get(name) or registry._LOADED.get(level, {}).get(name)


# -- config ----------------------------------------------------------------
def test_defaults_are_used_when_config_is_empty():
    config = Config(raw=dict(Config.__dataclass_fields__ and {}))
    assert config.use(registry.L4_VECTORSTORE) == "qdrant"
    assert config.use(registry.L3_FRAMEWORK) == "langgraph"


def test_env_override_swaps_a_level(monkeypatch):
    monkeypatch.setenv("ARC_L4_VECTORSTORE", "chroma")
    config = Config.load()
    assert config.use(registry.L4_VECTORSTORE) == "chroma"


def test_env_override_sets_a_setting(monkeypatch):
    monkeypatch.setenv("ARC_L0_INFERENCE__MODEL", "qwen2.5:3b")
    config = Config.load()
    assert config.settings(registry.L0_INFERENCE)["model"] == "qwen2.5:3b"


def test_env_override_reaches_pipeline_knobs(monkeypatch):
    monkeypatch.setenv("ARC_PIPELINE__TOP_K", "9")
    assert Config.load().pipeline["top_k"] == 9


def test_env_override_beats_the_adapter_specific_block():
    """Regression: env used to be merged UNDER the adapter block, not over it.

    `_apply_env` wrote into the level's common `settings`, and `settings()` then
    merged `l4_vectorstore.qdrant` on top of it -- so config.yaml's `localhost`
    silently won over an ARC_L4_VECTORSTORE__URL set in a compose file, and a
    containerised deployment pointed at the wrong host with no error anywhere.
    """
    raw = {
        "l4_vectorstore": {
            "use": "qdrant",
            "settings": {"collection": "arc_rector"},
            "qdrant": {"url": "http://localhost:6333"},
        }
    }
    from_yaml_only = Config(raw=raw)
    assert from_yaml_only.settings(registry.L4_VECTORSTORE)["url"] == "http://localhost:6333"

    overridden = Config(raw=raw, env={"l4_vectorstore": {"url": "http://qdrant:6333"}})
    assert overridden.settings(registry.L4_VECTORSTORE)["url"] == "http://qdrant:6333"
    # The rest of the block is untouched -- an override replaces one key, not the block.
    assert overridden.settings(registry.L4_VECTORSTORE)["collection"] == "arc_rector"


@pytest.mark.parametrize(
    "env_key,level,setting",
    [
        ("ARC_L4_VECTORSTORE__URL", registry.L4_VECTORSTORE, "url"),
        ("ARC_L5_EMBEDDINGS__BASE_URL", registry.L5_EMBEDDINGS, "base_url"),
    ],
)
def test_env_override_wins_against_the_real_config_yaml(monkeypatch, env_key, level, setting):
    """Same regression, end to end: the shipped config.yaml sets both of these
    in an adapter-specific block, which is exactly the case that used to lose."""
    baseline = Config.load().settings(level).get(setting)
    monkeypatch.setenv(env_key, "http://from-the-environment:1234")
    assert baseline != "http://from-the-environment:1234"
    assert Config.load().settings(level)[setting] == "http://from-the-environment:1234"


def test_unknown_env_level_is_ignored(monkeypatch):
    monkeypatch.setenv("ARC_L99_NOTHING", "x")
    Config.load()  # must not raise


def test_selection_covers_every_level():
    assert set(Config.load().selection()) == set(registry.LEVELS)


def test_config_yaml_names_only_real_adapters():
    """Guards against config.yaml drifting from the registry."""
    config = Config.load()
    for level in registry.LEVELS:
        assert config.use(level) in registry.known(level), (
            f"config.yaml sets {level}={config.use(level)}, which is not registered"
        )


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("False", False), ("42", 42), ("1.5", 1.5), ("none", None), ("text", "text")],
)
def test_env_values_are_coerced(raw, expected):
    assert _coerce(raw) == expected


def test_deep_merge_preserves_untouched_keys():
    merged = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}}
