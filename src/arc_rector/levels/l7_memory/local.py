"""L7 offline option: a local JSON fact store with lexical recall.

Mem0 is the default because it uses an LLM to decide what is worth remembering.
This adapter is the zero-dependency version of the same idea: it extracts
first-person statements with a small pattern set, stores them in a JSON file,
and recalls them by token overlap.

It is genuinely useful (the demo's cross-turn memory check runs green on it with
nothing installed) and genuinely limited: no semantic recall, no consolidation,
no contradiction handling. When "I live in Pune" is later replaced by "I moved to
Goa", Mem0 updates the fact and this store keeps both. That difference is the
argument for the L7 layer existing at all.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from ...interfaces import Memory
from ...types import MemoryRecord

# First-person statements worth keeping. Deliberately conservative: a false
# memory is worse than a missed one, because it silently poisons later answers.
# A fact ends at a sentence boundary or a clause joiner. Without those stops the
# character classes run greedily across the rest of the message and store a
# whole paragraph as one "fact".
_STOP = r"(?=[.!?,;]|\s+(?:and|but|so|because|then|while|which|what|who)\b|$)"

_FACT_PATTERNS = (
    # Case-sensitive on purpose: under IGNORECASE, [A-Z] also matches lowercase,
    # so "my name is Devanshu and" would capture the trailing "and" as a surname.
    re.compile(r"\b(?:[Mm]y name is|I am called|[Cc]all me)\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)?)"),
    re.compile(r"\bi\s+(?:live|work|am based)\s+in\s+([\w\s'\-]{2,60}?)" + _STOP, re.I),
    re.compile(
        r"\bi\s+(?:work|deal)\s+(?:mostly\s+|mainly\s+|primarily\s+|a lot\s+)?with\s+([\w\s'\-\.]{2,60}?)"
        + _STOP,
        re.I,
    ),
    re.compile(r"\bi\s+(?:am|'m)\s+(?:a|an)\s+([\w\s\-]{2,60}?)" + _STOP, re.I),
    re.compile(r"\bi\s+(?:prefer|like|love|use|want|need)\s+([\w\s'\-]{2,80}?)" + _STOP, re.I),
    re.compile(r"\bi\s+(?:don't|do not|dislike|hate|avoid)\s+([\w\s'\-]{2,80}?)" + _STOP, re.I),
    re.compile(r"\bmy\s+(\w+\s+is\s+[\w\s'\-]{2,60}?)" + _STOP, re.I),
)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from in is it its of on or the to with i my me".split())


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1}


class LocalMemory(Memory):
    name = "local"

    def __init__(self, *, path: str = ".arc_rector/memory", **_: Any) -> None:
        self.path = Path(path)
        self._file = self.path / "local_memory.json"
        self._data: dict[str, list[dict[str, Any]]] | None = None

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if self._data is None:
            if self._file.exists():
                try:
                    self._data = json.loads(self._file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data or {}, indent=2), encoding="utf-8")

    @staticmethod
    def _extract(messages: Sequence[dict[str, str]]) -> list[str]:
        facts: list[str] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = (message.get("content") or "").strip()
            for pattern in _FACT_PATTERNS:
                for match in pattern.finditer(content):
                    fact = match.group(0).strip().rstrip(".,!?")
                    if fact and fact not in facts:
                        facts.append(fact)
        return facts

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        facts = self._extract(messages)
        if not facts:
            return []
        store = self._load()
        bucket = store.setdefault(user_id, [])
        existing = {row["text"].lower() for row in bucket}
        added: list[str] = []
        for fact in facts:
            if fact.lower() in existing:
                continue
            bucket.append({"text": fact, "ts": time.time()})
            added.append(fact)
        if added:
            self._save()
        return added

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        bucket = self._load().get(user_id, [])
        if not bucket:
            return []
        query_tokens = _tokens(query)
        scored: list[MemoryRecord] = []
        for row in bucket:
            text = row["text"]
            overlap = query_tokens & _tokens(text)
            # Recent facts win ties, so an updated preference surfaces first.
            score = len(overlap) + (row.get("ts", 0.0) / 1e12)
            if overlap:
                scored.append(MemoryRecord(text=text, memory_id=text[:24], score=score))
        scored.sort(key=lambda r: -r.score)
        if scored:
            return scored[:top_k]
        # No lexical hit: fall back to the most recent facts, which is usually
        # better than returning nothing for short questions like "where am I?".
        recent = sorted(bucket, key=lambda r: -r.get("ts", 0.0))[:top_k]
        return [MemoryRecord(text=r["text"], memory_id=r["text"][:24], score=0.0) for r in recent]

    def reset(self, user_id: str = "") -> None:
        store = self._load()
        if user_id:
            store.pop(user_id, None)
        else:
            store.clear()
        self._save()

    def all_facts(self, user_id: str) -> list[str]:
        return [row["text"] for row in self._load().get(user_id, [])]


class NoMemory(Memory):
    """L7 disabled. Explicit rather than implicit."""

    name = "none"

    def __init__(self, **_: Any) -> None:
        pass

    def add(self, messages: Sequence[dict[str, str]], user_id: str) -> list[str]:
        return []

    def search(self, query: str, user_id: str, top_k: int = 5) -> list[MemoryRecord]:
        return []

    def reset(self, user_id: str = "") -> None:
        pass
