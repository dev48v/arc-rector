"""L7 alternative: Zep -- a temporal knowledge graph instead of a fact list.

What Zep buys you over a flat fact store is *time*. Every fact it extracts is an
edge with `valid_at` and `invalid_at`, so "Devanshu works at InXpress" does not
silently coexist with "Devanshu works at Acme" -- the earlier edge is invalidated
and the graph knows which one holds now. Retrieval is hybrid (semantic + BM25 +
graph traversal) with a reranker, and it returns facts rather than raw chat
turns, which is exactly the shape a prompt wants.

Read this part before you configure it, because the brief this adapter was
written from was out of date and the packaging is genuinely confusing:

  * Zep **Community Edition -- the self-hostable server -- is discontinued.**
    Its code sits in a legacy folder of the getzep/zep repo. There is no
    maintained `docker compose up` that gives you a community Zep in 2026.
  * `pip install zep-python` still resolves, but it pins **2.0.2 (Sep 2024)** and
    its own README says the OSS-compatible SDK lives on an unmaintained `oss`
    branch. Do not build on it.
  * The maintained client is **`zep-cloud`**, and that is what this adapter
    imports. Despite the name it is not hardwired to the cloud: the generated
    `Zep` client takes an explicit `base_url`, so it will talk to any host
    speaking the Zep API -- an archived CE deployment, an internal proxy, or
    `api.getzep.com`.

So `base_url` still defaults to `http://localhost:8000` to keep the self-hosted
posture, and this adapter will work against anything answering there. But be
clear-eyed: on a stock 2026 machine nothing is listening on that port, and the
only fully self-hostable option in this repo with the same temporal-graph model
is the `graphiti` adapter -- Graphiti is the open-source engine underneath Zep.

Gotcha that will look like a bug: fact extraction is **asynchronous**.
`graph.add` returns as soon as the episode is queued, and the facts it produces
are not searchable for a few seconds. A test that writes then immediately reads
will get an empty list. `add()` therefore returns the message texts it submitted,
not the extracted facts -- those do not exist yet.

The API key is read from the `api_key` argument or the ZEP_API_KEY environment
variable and is never printed, logged, or included in an error message.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ...interfaces import Memory
from ...registry import require
from ...types import MemoryRecord


class ZepMemory(Memory):
    name = "zep"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        data_type: str = "message",
        timeout: float = 60.0,
        auto_create_user: bool = True,
        min_fact_rating: float = 0.0,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Env fallback only; never rendered anywhere.
        self._api_key = api_key or os.getenv("ZEP_API_KEY") or ""
        self.data_type = data_type
        self.timeout = timeout
        self.auto_create_user = auto_create_user
        self.min_fact_rating = min_fact_rating
        self._c: Any = None
        self._known_users: set[str] = set()

    @property
    def _client(self) -> Any:
        """Connect lazily so merely selecting this adapter never fails."""
        if self._c is None:
            zep = require("zep-cloud", "zep", "zep_cloud.client")
            self._c = zep.Zep(
                base_url=self.base_url,
                api_key=self._api_key or None,
                timeout=self.timeout,
            )
        return self._c

    def _ensure_user(self, user_id: str) -> None:
        """Zep rejects graph writes for an unknown user; creating twice is a 409."""
        if not self.auto_create_user or not user_id or user_id in self._known_users:
            return
        try:
            self._client.user.add(user_id=user_id)
        except Exception:
            # Already exists, or the server disallows it. Either way, carry on.
            pass
        self._known_users.add(user_id)

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        if not messages:
            return []
        self._ensure_user(user_id)

        stored: list[str] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = str(message.get("role", "user"))
            # type="message" wants "role: content"; type="text" takes the body as-is.
            data = f"{role}: {content}" if self.data_type == "message" else content
            self._client.graph.add(data=data, type=self.data_type, user_id=user_id)
            stored.append(content)
        return stored

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        if not query:
            return []
        results = self._client.graph.search(
            query=query,
            user_id=user_id,
            limit=max(1, min(top_k, 50)),
        )
        edges = getattr(results, "edges", None) or []
        out: list[MemoryRecord] = []
        for edge in edges:
            fact = str(getattr(edge, "fact", "") or "").strip()
            if not fact:
                continue
            score = float(getattr(edge, "score", None) or getattr(edge, "relevance", None) or 0.0)
            if score < self.min_fact_rating:
                continue
            out.append(
                MemoryRecord(
                    text=fact,
                    # The wire field is "uuid"; the generated model aliases it to uuid_.
                    memory_id=str(getattr(edge, "uuid_", "") or getattr(edge, "uuid", "") or ""),
                    score=score,
                )
            )
        return out[:top_k]

    def reset(self, user_id: str = "") -> None:
        """Deleting the user deletes their graph -- there is no per-fact wipe."""
        if user_id:
            try:
                self._client.user.delete(user_id)
            except Exception:
                pass
            self._known_users.discard(user_id)
            return

        # Global reset: page through the users this project can see.
        try:
            page = self._client.user.list_ordered(page_size=100)
            for user in getattr(page, "users", None) or []:
                uid = getattr(user, "user_id", "")
                if uid:
                    self._client.user.delete(uid)
        except Exception:
            pass
        self._known_users.clear()
