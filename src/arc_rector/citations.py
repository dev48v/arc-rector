"""Citation numbering, formatting, and validation.

Kept separate from any framework so it is deterministic and unit-testable: the
same retrieved chunks always produce the same numbered context block and the
same source list. An answer without working citations is not a RAG answer, so
this module also verifies that the markers a model emitted actually exist.
"""

from __future__ import annotations

import re
from typing import Sequence

from .types import Citation, Retrieved

MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def build_citations(hits: Sequence[Retrieved]) -> list[Citation]:
    """Number the retrieved chunks 1..n, deduplicating repeats of one chunk."""
    citations: list[Citation] = []
    seen: dict[str, int] = {}
    for hit in hits:
        chunk = hit.chunk
        if chunk.chunk_id in seen:
            continue
        marker = len(citations) + 1
        seen[chunk.chunk_id] = marker
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                title=chunk.title or chunk.doc_id,
                source=chunk.source,
                quote=chunk.text[:200],
            )
        )
    return citations


def format_context(hits: Sequence[Retrieved], max_chars: int = 6000) -> tuple[str, list[Citation]]:
    """Render retrieved chunks as a numbered context block for the prompt.

    Returns the block and the citations that actually fitted inside it -- a
    citation is only offered to the model if its text survived truncation.
    """
    citations = build_citations(hits)
    by_id = {c.chunk_id: c for c in citations}

    blocks: list[str] = []
    used: list[Citation] = []
    budget = max_chars
    for hit in hits:
        cite = by_id.get(hit.chunk.chunk_id)
        if cite is None or any(u.chunk_id == cite.chunk_id for u in used):
            continue
        header = f"[{cite.marker}] {cite.title}"
        if cite.source:
            header += f" -- {cite.source}"
        body = hit.chunk.text.strip()
        entry = f"{header}\n{body}"
        if len(entry) > budget:
            if budget < 200:
                break
            entry = entry[:budget].rstrip() + " ..."
        blocks.append(entry)
        used.append(cite)
        budget -= len(entry)
        if budget <= 0:
            break

    # Renumber so markers are always contiguous from 1, even after truncation.
    renumbered: list[Citation] = []
    remap: dict[int, int] = {}
    for new_marker, cite in enumerate(used, start=1):
        remap[cite.marker] = new_marker
        renumbered.append(
            Citation(
                marker=new_marker,
                chunk_id=cite.chunk_id,
                title=cite.title,
                source=cite.source,
                quote=cite.quote,
            )
        )
    text = "\n\n".join(
        re.sub(r"^\[(\d+)\]", lambda m: f"[{remap[int(m.group(1))]}]", block)
        for block in blocks
    )
    return text, renumbered


def used_markers(answer: str) -> list[int]:
    """Every [n] marker the model actually emitted, in order, deduplicated."""
    out: list[int] = []
    for match in MARKER_RE.finditer(answer):
        num = int(match.group(1))
        if num not in out:
            out.append(num)
    return out


def prune_citations(answer: str, citations: Sequence[Citation]) -> list[Citation]:
    """Keep only the citations the answer actually referenced.

    Listing sources the model never used is how RAG demos overstate their
    grounding, so the default is to show only what was cited. If the model cited
    nothing, fall back to showing everything that was in context.
    """
    markers = set(used_markers(answer))
    if not markers:
        return list(citations)
    return [c for c in citations if c.marker in markers]


def has_valid_citations(answer: str, citations: Sequence[Citation]) -> bool:
    """True when the answer cites at least one marker that really exists."""
    valid = {c.marker for c in citations}
    return bool(valid & set(used_markers(answer)))


def dangling_markers(answer: str, citations: Sequence[Citation]) -> list[int]:
    """Markers the model invented that have no matching source."""
    valid = {c.marker for c in citations}
    return [m for m in used_markers(answer) if m not in valid]
