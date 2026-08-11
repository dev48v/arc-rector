"""L7 alternative: Letta (Apache-2.0), formerly MemGPT.

What Letta buys you is a different idea of memory. Mem0 and Zep are stores you
write to and query. Letta is a stateful *agent* that owns its own memory and
edits it: core-memory blocks live permanently in the context window and the
agent rewrites them with tool calls, while overflow goes to archival memory that
the agent searches on its own initiative. Nothing outside decides what is worth
remembering -- the agent does, mid-conversation. That is the whole thesis, and
for a long-running assistant it is a genuinely better model than a fact table.

Self-hosts cleanly: `docker run letta/letta:latest`, default port 8283, and the
`base_url` argument below points straight at it. No key needed locally.

**Where the impedance mismatch is, honestly.** Arc Rector's `Memory` interface
is a store: `add(messages, user_id)` writes, `search(query, user_id)` reads, and
`user_id` is the only scope. Letta has no user-scoped fact store to bind that to,
so this adapter makes two compromises you should know about:

  1. `user_id` is mapped onto **a dedicated agent per user**, created on first
     use and looked up by name after that. Users are not a Letta concept; agents
     are. So "reset a user" means "delete an agent", and each user costs a
     server-side agent with its own context window.
  2. `add()` writes straight into archival memory via the passages API. This
     **bypasses the agent entirely** -- no reasoning step, no decision about
     salience, no core-memory update. It is a plain insert into the agent's
     archival store. Which means that in this configuration you are paying for
     Letta's architecture and using it as a vector store. The part that makes
     Letta worth choosing -- the agent deciding what to remember -- happens in
     `client.agents.messages.create(...)`, which the `Memory` contract has
     nowhere to call from. If Letta is the layer you actually want, drive it from
     L3 as the framework, not from L7 as a store.

Gotcha at setup: creating an agent requires the server to have a model and an
embedding model configured. A bare `docker run` with no provider key configured
will accept the connection and then fail agent creation -- so the first error you
see from this adapter is usually about models, not about memory. Point the Letta
server at your local Ollama to keep the stack zero-key.

Any API key comes from the `api_key` argument or LETTA_API_KEY and is never
printed, logged, or included in an error message.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ...interfaces import Memory
from ...registry import require
from ...types import MemoryRecord


class LettaMemory(Memory):
    name = "letta"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8283",
        api_key: str | None = None,
        agent_prefix: str = "arc-rector",
        model: str | None = None,
        embedding: str | None = None,
        timeout: float = 120.0,
        tag: str = "arc-rector",
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Env fallback only; never rendered anywhere.
        self._api_key = api_key or os.getenv("LETTA_API_KEY") or ""
        self.agent_prefix = agent_prefix
        self.model = model
        self.embedding = embedding
        self.timeout = timeout
        self.tag = tag
        self._c: Any = None
        self._agents: dict[str, str] = {}

    @property
    def _client(self) -> Any:
        """Connect lazily so merely selecting this adapter never fails."""
        if self._c is None:
            letta = require("letta-client", "letta", "letta_client")
            self._c = letta.Letta(
                base_url=self.base_url,
                api_key=self._api_key or None,
                timeout=self.timeout,
            )
        return self._c

    def _agent_name(self, user_id: str) -> str:
        return f"{self.agent_prefix}-{user_id or 'default'}"

    def _agent_id(self, user_id: str) -> str:
        """One agent per user_id: find it by name, else create it."""
        cached = self._agents.get(user_id)
        if cached:
            return cached

        name = self._agent_name(user_id)
        for agent in self._client.agents.list(name=name, limit=1):
            agent_id = str(getattr(agent, "id", ""))
            if agent_id:
                self._agents[user_id] = agent_id
                return agent_id

        create_kwargs: dict[str, Any] = {"name": name, "tags": [self.tag]}
        if self.model:
            create_kwargs["model"] = self.model
        if self.embedding:
            create_kwargs["embedding"] = self.embedding
        agent = self._client.agents.create(**create_kwargs)
        agent_id = str(getattr(agent, "id", ""))
        self._agents[user_id] = agent_id
        return agent_id

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        if not messages:
            return []
        agent_id = self._agent_id(user_id)

        stored: list[str] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = str(message.get("role", "user"))
            # POST /v1/agents/{id}/archival-memory -- a direct write, no agent turn.
            self._client.agents.passages.create(
                agent_id,
                text=f"{role}: {content}",
                tags=[self.tag, f"user:{user_id}"] if user_id else [self.tag],
            )
            stored.append(content)
        return stored

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        if not query:
            return []
        agent_id = self._agent_id(user_id)
        response = self._client.agents.passages.search(agent_id, query=query, top_k=top_k)

        out: list[MemoryRecord] = []
        for passage in _iter_results(response):
            text = str(_field(passage, "text") or _field(passage, "content") or "").strip()
            if not text:
                continue
            raw_score = _field(passage, "score")
            out.append(
                MemoryRecord(
                    text=text,
                    memory_id=str(_field(passage, "id") or ""),
                    score=float(raw_score) if raw_score is not None else 0.0,
                )
            )
        return out[:top_k]

    def reset(self, user_id: str = "") -> None:
        """Deleting the agent deletes its archival memory with it."""
        if user_id:
            agent_id = self._agents.pop(user_id, "") or self._lookup(self._agent_name(user_id))
            if agent_id:
                self._client.agents.delete(agent_id)
            return

        # Global reset: every agent this adapter created carries our tag.
        for agent in self._client.agents.list(tags=[self.tag]):
            agent_id = str(getattr(agent, "id", ""))
            if agent_id:
                self._client.agents.delete(agent_id)
        self._agents.clear()

    def _lookup(self, name: str) -> str:
        for agent in self._client.agents.list(name=name, limit=1):
            return str(getattr(agent, "id", ""))
        return ""


def _iter_results(response: Any) -> list[Any]:
    """Search returns a list-like page; older builds wrapped it in .results."""
    for attr in ("results", "passages", "data", "items"):
        value = getattr(response, attr, None)
        if isinstance(value, list):
            return value
    if isinstance(response, list):
        return response
    try:
        return list(response)
    except TypeError:
        return []


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
