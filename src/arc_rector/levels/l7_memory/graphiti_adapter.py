"""L7 alternative: Graphiti (Apache-2.0) -- the temporal graph engine under Zep.

What Graphiti buys you is bi-temporal memory, and it is the most interesting idea
at this level. Every relationship it extracts is an edge carrying four timestamps:
when the fact became true in the world (`valid_at` / `invalid_at`) and when the
system learned it (`created_at` / `expired_at`). New information does not
overwrite old information -- it *invalidates* it, and the superseded edge stays
in the graph. So you can ask what is true now, and you can also ask what you
believed last March and when you stopped believing it. A flat fact store cannot
answer the second question at all.

It also updates incrementally. Adding an episode does not trigger a batch
re-derivation of the graph the way a rebuild-the-index approach would, which is
what makes it viable for live conversation rather than nightly ingestion.

**The gotcha is the infrastructure, and it is why this is not the default.**
Graphiti is a library, not a server: it needs a graph database you provision
yourself -- Neo4j (default, `bolt://localhost:7687`) or FalkorDB via the
`graphiti-core[falkordb]` extra. That is a JVM container with its own memory
budget and its own auth, versus `mem0` which writes to a local file. It also
defaults to OpenAI for both extraction and embeddings, so a zero-key setup means
passing your own `llm_client` and `embedder` (there are Ollama-compatible ones)
before this adapter is genuinely keyless. Two moving parts to stand up before the
first fact is stored is a real cost for a teaching stack.

Second gotcha, the one that produces a baffling runtime error: the API is async
and the Neo4j async driver **pins itself to the event loop it was constructed
on**. Calling `asyncio.run(...)` per method creates a *new* loop each time, so a
cached client works on the first call and then fails with "attached to a
different loop" on the second. This adapter therefore owns one background event
loop for its whole lifetime and submits every coroutine to it, rather than
wrapping each call in a bare `asyncio.run`.

Extraction is also an LLM workload, so `add()` is seconds, not milliseconds, and
facts are only searchable once extraction finishes.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, Sequence

from ...interfaces import Memory
from ...registry import require
from ...types import MemoryRecord


class _BackgroundLoop:
    """One long-lived event loop on its own thread, so the driver stays valid."""

    def __init__(self) -> None:
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self, coro: Any) -> Any:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever, name="graphiti-loop", daemon=True
                )
                self._thread.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def stop(self) -> None:
        with self._lock:
            if self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None


class GraphitiMemory(Memory):
    name = "graphiti"

    def __init__(
        self,
        *,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        group_prefix: str = "arc-rector",
        source_description: str = "arc-rector conversation",
        build_indices: bool = True,
        **_: Any,
    ) -> None:
        self.uri = uri
        self.user = user
        # Held only to construct the driver; never rendered anywhere.
        self._password = password
        self.group_prefix = group_prefix
        self.source_description = source_description
        self.build_indices = build_indices
        self._g: Any = None
        self._nodes: Any = None
        self._loop = _BackgroundLoop()
        self._indices_built = False

    @property
    def _client(self) -> Any:
        """Connect lazily so merely selecting this adapter never fails."""
        if self._g is None:
            core = require("graphiti-core", "graphiti", "graphiti_core")
            self._nodes = require("graphiti-core", "graphiti", "graphiti_core.nodes")
            # Constructed inside the background loop so the driver binds to it.
            self._g = self._loop.run(self._construct(core))
            if self.build_indices and not self._indices_built:
                self._loop.run(self._g.build_indices_and_constraints())
                self._indices_built = True
        return self._g

    async def _construct(self, core: Any) -> Any:
        return core.Graphiti(self.uri, self.user, self._password)

    def _group_id(self, user_id: str) -> str:
        return f"{self.group_prefix}-{user_id or 'default'}"

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        if not messages:
            return []
        client = self._client
        episode_type = self._nodes.EpisodeType
        group_id = self._group_id(user_id)
        now = datetime.now(timezone.utc)

        stored: list[str] = []

        async def _write() -> None:
            for index, message in enumerate(messages):
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                role = str(message.get("role", "user"))
                # EpisodeType.message expects the body formatted as "actor: content".
                await client.add_episode(
                    name=f"{group_id}-{now.isoformat()}-{index}",
                    episode_body=f"{role}: {content}",
                    source=episode_type.message,
                    source_description=self.source_description,
                    reference_time=now,
                    group_id=group_id,
                )
                stored.append(content)

        self._loop.run(_write())
        return stored

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        if not query:
            return []
        client = self._client
        edges = self._loop.run(
            client.search(query, group_ids=[self._group_id(user_id)], num_results=top_k)
        )

        out: list[MemoryRecord] = []
        for edge in edges or []:
            fact = str(getattr(edge, "fact", "") or "").strip()
            if not fact:
                continue
            score = getattr(edge, "score", None)
            out.append(
                MemoryRecord(
                    text=fact,
                    memory_id=str(getattr(edge, "uuid", "") or ""),
                    score=float(score) if score is not None else 0.0,
                )
            )
        return out[:top_k]

    def reset(self, user_id: str = "") -> None:
        """clear_data drops nodes and edges; an empty user_id clears every group."""
        client = self._client
        ops = require(
            "graphiti-core",
            "graphiti",
            "graphiti_core.utils.maintenance.graph_data_operations",
        )
        group_ids = [self._group_id(user_id)] if user_id else None
        self._loop.run(ops.clear_data(client.driver, group_ids))

    def close(self) -> None:
        """Release the Neo4j driver and stop the background loop."""
        if self._g is not None:
            try:
                self._loop.run(self._g.close())
            except Exception:
                pass
            self._g = None
        self._loop.stop()
