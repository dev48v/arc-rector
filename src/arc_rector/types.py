"""Core data types shared by every level of the stack.

These are deliberately plain dataclasses with no third-party dependencies so that
an adapter for any level can be written without pulling in the rest of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A parsed source document, before chunking."""

    doc_id: str
    text: str
    source: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a Document."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = 0
    source: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Flat dict stored alongside the vector in whichever vector DB is active."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "source": self.source,
            "title": self.title,
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        known = {"chunk_id", "doc_id", "text", "ordinal", "source", "title"}
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            text=str(payload.get("text", "")),
            ordinal=int(payload.get("ordinal", 0) or 0),
            source=str(payload.get("source", "")),
            title=str(payload.get("title", "")),
            metadata={k: v for k, v in payload.items() if k not in known},
        )


@dataclass
class Retrieved:
    """A chunk returned by a vector store, with its similarity score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Citation:
    """A numbered reference attached to an answer."""

    marker: int
    chunk_id: str
    title: str
    source: str
    quote: str = ""

    def render(self) -> str:
        label = self.title or self.source or self.chunk_id
        if self.source and self.source != label:
            return f"[{self.marker}] {label} ({self.source})"
        return f"[{self.marker}] {label}"


@dataclass
class Answer:
    """The end-to-end result of a query."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    memories_used: list[str] = field(default_factory=list)
    trace_id: str = ""

    def render(self) -> str:
        if self.blocked:
            return f"[BLOCKED] {self.block_reason}"
        out = [self.text]
        if self.citations:
            out.append("")
            out.append("Sources:")
            out.extend(c.render() for c in self.citations)
        return "\n".join(out)


@dataclass
class GuardResult:
    """Outcome of an input or output guardrail check."""

    allowed: bool
    text: str = ""
    reason: str = ""
    validator: str = ""


@dataclass
class MemoryRecord:
    """A single remembered fact about a user."""

    text: str
    memory_id: str = ""
    score: float = 0.0
