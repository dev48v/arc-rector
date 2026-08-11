"""L7 default: Mem0 (Apache-2.0).

Mem0 is the default because it does the thing a JSON fact list cannot: it uses an
LLM to decide what in a conversation is worth remembering, and it *reconciles*
new facts against old ones. Tell it "I live in Pune" and later "I moved to Goa"
and it updates rather than accumulating a contradiction.

The zero-key catch, and why this adapter is longer than you would expect: Mem0
defaults to OpenAI for both its LLM and its embedder, so out of the box it wants
an API key. This adapter reconfigures it onto the same local Ollama models the
rest of the stack uses, and onto a local Qdrant collection, so it stays within
the project's no-key rule. If Mem0 or its backing services are unreachable it
falls back to `LocalMemory` (when `fallback_to_local` is on) rather than taking
the whole turn down -- memory is an enhancement, not a hard dependency.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Sequence, TypeVar

from ...interfaces import Memory
from ...registry import require
from ...types import MemoryRecord
from .local import LocalMemory

T = TypeVar("T")


class Mem0Memory(Memory):
    name = "mem0"

    def __init__(
        self,
        *,
        path: str = ".arc_rector/memory",
        llm_model: str = "llama3.2:3b",
        embed_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        qdrant_url: str = "http://localhost:6333",
        collection: str = "arc_rector_memory",
        embed_dims: int = 768,
        fallback_to_local: bool = True,
        timeout: float = 45.0,
        telemetry: bool = False,
        **_: Any,
    ) -> None:
        self.path = path
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.ollama_url = ollama_url
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.embed_dims = embed_dims
        self.fallback_to_local = fallback_to_local
        self.timeout = timeout
        self.telemetry = telemetry
        self._memory: Any = None
        self._fallback: LocalMemory | None = None
        self._failed = False
        self.active_backend = "mem0"

    def _guard(self, call: Callable[[], T], default: Callable[[], T]) -> T:
        """Run a mem0 call under a wall-clock budget, else use the fallback.

        Mem0's `add` is an LLM round trip (extract facts, then reconcile them
        against existing ones). On a CPU-only box that is minutes, not seconds,
        and a demo that appears to hang is worse than a demo that degrades. A
        hang is not an exception, so try/except cannot catch it -- hence the
        watchdog.

        Each call gets its OWN daemon thread rather than a shared pool. With a
        single-worker pool a timed-out call keeps running and every later call
        queues behind it, so one slow turn makes all subsequent turns wait for
        the stuck one *plus* their own timeout -- head-of-line blocking that
        turns a 150s budget into a compounding stall. A daemon thread is also
        never joined at exit, so a wedged mem0 call cannot stop the process
        from exiting. The thread is left to finish rather than being killed:
        killing a thread mid-write is how a vector collection gets corrupted.
        """
        box: dict[str, Any] = {}

        def runner() -> None:
            try:
                box["value"] = call()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc

        worker = threading.Thread(target=runner, name="mem0-call", daemon=True)
        worker.start()
        worker.join(self.timeout)

        if worker.is_alive():
            self.active_backend = f"local (mem0 exceeded {self.timeout:g}s)"
            if not self.fallback_to_local:
                raise TimeoutError(f"mem0 call exceeded {self.timeout:g}s")
            return default()
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box["value"]  # type: ignore[return-value]

    def _config(self) -> dict[str, Any]:
        """Point every Mem0 subsystem at local, self-hosted services."""
        host, _, port = self.qdrant_url.replace("http://", "").replace("https://", "").partition(":")
        return {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.llm_model,
                    "ollama_base_url": self.ollama_url,
                    "temperature": 0.1,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.embed_model,
                    "ollama_base_url": self.ollama_url,
                    "embedding_dims": self.embed_dims,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection,
                    "host": host or "localhost",
                    "port": int(port or 6333),
                    "embedding_model_dims": self.embed_dims,
                },
            },
        }

    @property
    def _client(self) -> Any:
        if self._memory is None and not self._failed:
            try:
                # Set before import: mem0 reads this at module load to decide
                # whether to start a PostHog client. A stack that advertises
                # "no vendor accounts" should not phone home by default.
                if not self.telemetry:
                    os.environ.setdefault("MEM0_TELEMETRY", "False")
                mem0 = require("mem0ai", "mem0", "mem0")
                self._memory = mem0.Memory.from_config(self._config())
            except Exception:
                self._failed = True
                self._memory = None
        return self._memory

    def _local(self) -> LocalMemory:
        if self._fallback is None:
            self._fallback = LocalMemory(path=self.path)
        self.active_backend = "local (mem0 unavailable)"
        return self._fallback

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        client = self._client
        if client is None:
            if not self.fallback_to_local:
                raise RuntimeError("Mem0 is unavailable and fallback_to_local is disabled")
            return self._local().add(messages, user_id)

        def run() -> list[str]:
            return _extract_texts(client.add(list(messages), user_id=user_id))

        def fallback() -> list[str]:
            return self._local().add(messages, user_id)

        try:
            return self._guard(run, fallback)
        except Exception:
            if not self.fallback_to_local:
                raise
            return fallback()

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        client = self._client
        if client is None:
            if not self.fallback_to_local:
                raise RuntimeError("Mem0 is unavailable and fallback_to_local is disabled")
            return self._local().search(query, user_id, top_k)

        def run() -> list[MemoryRecord]:
            result = client.search(query=query, user_id=user_id, limit=top_k)
            rows = result.get("results", result) if isinstance(result, dict) else result
            out: list[MemoryRecord] = []
            for row in rows or []:
                if isinstance(row, dict):
                    text = row.get("memory") or row.get("text") or ""
                    if text:
                        out.append(
                            MemoryRecord(
                                text=str(text),
                                memory_id=str(row.get("id", "")),
                                score=float(row.get("score") or 0.0),
                            )
                        )
            return out[:top_k]

        def fallback() -> list[MemoryRecord]:
            return self._local().search(query, user_id, top_k)

        try:
            return self._guard(run, fallback)
        except Exception:
            if not self.fallback_to_local:
                raise
            return fallback()

    def reset(self, user_id: str = "") -> None:
        client = self._client
        if client is not None:
            try:
                if user_id:
                    client.delete_all(user_id=user_id)
                else:
                    client.reset()
            except Exception:
                pass
        if self._fallback is not None or self.fallback_to_local:
            self._local().reset(user_id)


def _extract_texts(result: Any) -> list[str]:
    """Mem0 has returned both a bare list and {'results': [...]} across versions."""
    rows = result.get("results", []) if isinstance(result, dict) else (result or [])
    texts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            text = row.get("memory") or row.get("text") or row.get("data")
            if text:
                texts.append(str(text))
    return texts
