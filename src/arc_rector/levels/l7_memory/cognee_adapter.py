"""L7 alternative: Cognee (Apache-2.0) -- memory as a knowledge graph.

What Cognee buys you is structure. Mem0 stores facts as independent strings;
Cognee runs an ECL pipeline (Extract, Cognify, Load) that pulls entities and
relationships out of your text and builds an actual **graph**, backed by a graph
store plus a vector store. The payoff is multi-hop recall: "who else worked on
the thing Barnali mentioned" is a traversal, not a similarity search that happens
to get lucky. Point it at Kuzu/NetworkX and LanceDB and it runs entirely on
local files, no server.

The API is async top to bottom, so every call here is wrapped. `_run` is a little
more careful than a bare `asyncio.run`: that raises if it is called from inside
an already-running event loop, which is exactly what happens when a sync adapter
gets invoked from an async web handler. When a loop is already running the
coroutine is handed to a fresh loop on its own thread instead.

**The gotcha, and it is a real cost: Cognee wants an LLM, and not at query time
-- at write time.** `cognify()` is what turns raw text into the graph, and it
does that by prompting a model to extract entities and edges. So every `add()`
here is an LLM workload, not a database insert. Practical consequences:

  * It is slow. Seconds per document, not milliseconds. This adapter batches a
    whole turn into one `add` + one `cognify` for that reason.
  * It needs `LLM_API_KEY` (and embedding config) set before it will work.
    Cognee reads its own environment/config, not Arc Rector's L0 setting -- so
    swapping L0 to Ollama does NOT swap Cognee's model. To stay zero-key, point
    Cognee at your local Ollama via its own config (LLM_PROVIDER=ollama,
    LLM_ENDPOINT=http://localhost:11434/v1) before selecting this adapter.
  * If the LLM call fails, `add()` fails. There is no store-now-extract-later.

Search type matters too. Cognee's own default is `GRAPH_COMPLETION`, which is
RAG -- it calls an LLM and returns a written *answer*. That is the wrong shape
for `Memory.search`, which is supposed to return facts to put in a prompt, and
it would mean a second LLM call on every turn. This adapter defaults to `CHUNKS`,
which returns retrieved text with no generation. Set `search_type="GRAPH_
COMPLETION"` if you want the graph to answer rather than to recall.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import Any, Sequence

from ...interfaces import Memory
from ...registry import require
from ...types import MemoryRecord

_SAFE = re.compile(r"[^0-9a-zA-Z_]+")


def _run(coro: Any) -> Any:
    """asyncio.run, but survives being called from inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised on the calling thread
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


class CogneeMemory(Memory):
    name = "cognee"

    def __init__(
        self,
        *,
        dataset_prefix: str = "arc_rector",
        search_type: str = "CHUNKS",
        cognify_on_add: bool = True,
        top_k: int = 5,
        **_: Any,
    ) -> None:
        self.dataset_prefix = dataset_prefix
        self.search_type = search_type
        self.cognify_on_add = cognify_on_add
        self.top_k = top_k
        self._lib: Any = None

    @property
    def _cognee(self) -> Any:
        """Import lazily: selecting this adapter must not require the install."""
        if self._lib is None:
            self._lib = require("cognee", "cognee")
        return self._lib

    def _dataset(self, user_id: str) -> str:
        """Dataset names become table/graph identifiers, so keep them tame."""
        return f"{self.dataset_prefix}_{_SAFE.sub('_', user_id) or 'default'}"

    def _search_type(self) -> Any:
        enum = getattr(self._cognee, "SearchType")
        try:
            return enum[self.search_type.upper()]
        except KeyError:
            valid = ", ".join(sorted(member.name for member in enum))
            raise ValueError(f"Unknown cognee search_type '{self.search_type}'. Valid: {valid}")

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        if not messages:
            return []
        cognee = self._cognee
        dataset = self._dataset(user_id)

        texts: list[str] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if content:
                texts.append(f"{message.get('role', 'user')}: {content}")
        if not texts:
            return []

        async def _write() -> None:
            await cognee.add("\n".join(texts), dataset_name=dataset)
            if self.cognify_on_add:
                # This is the LLM step. Everything above is just staging.
                await cognee.cognify(datasets=[dataset])

        _run(_write())
        return [text.split(": ", 1)[-1] for text in texts]

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        if not query:
            return []
        cognee = self._cognee
        dataset = self._dataset(user_id)
        limit = top_k or self.top_k
        query_type = self._search_type()

        async def _read() -> Any:
            return await cognee.search(
                query_text=query,
                query_type=query_type,
                datasets=[dataset],
                top_k=limit,
            )

        results = _run(_read()) or []
        out: list[MemoryRecord] = []
        for item in results:
            text = _as_text(item)
            if not text:
                continue
            score = _field(item, "score")
            out.append(
                MemoryRecord(
                    text=text,
                    memory_id=str(_field(item, "id") or ""),
                    score=float(score) if score is not None else 0.0,
                )
            )
        return out[:limit]

    def reset(self, user_id: str = "") -> None:
        """Cognee prunes globally; there is no delete-one-dataset shortcut here."""
        cognee = self._cognee

        async def _wipe() -> None:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

        _run(_wipe())


def _as_text(item: Any) -> str:
    """Results are SearchResult models for some types, plain strings for others."""
    if isinstance(item, str):
        return item.strip()
    for key in ("text", "content", "answer", "chunk", "name", "description"):
        value = _field(item, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(item).strip()


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
