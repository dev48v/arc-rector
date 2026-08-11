"""Deterministic text chunking.

Deliberately dependency-free and fully deterministic so it can be unit-tested
without a model, a network, or a container. Chunking is where most RAG quality
is won or lost, so it is worth keeping legible.

Strategy: split on paragraph boundaries, greedily pack paragraphs into chunks of
at most `chunk_size` characters, then carry `chunk_overlap` characters of tail
context into the next chunk so a fact spanning a boundary is still retrievable.
A single paragraph longer than `chunk_size` is hard-split on sentence ends, and
only then on raw character count.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .types import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    """Collapse Windows newlines and runs of blank lines; strip trailing spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_long(block: str, limit: int) -> list[str]:
    """Break an over-long paragraph on sentence ends, then on characters."""
    if len(block) <= limit:
        return [block]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(block):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        # A single sentence longer than the limit: hard-split it.
        while len(sentence) > limit:
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        current = sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def _overlap_tail(text: str, overlap: int) -> str:
    """Take the last `overlap` chars, snapped forward to a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    """Split `text` into overlapping chunks. Pure function -- easy to test."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalise(text)
    if not text:
        return []

    blocks: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if para:
            blocks.extend(_split_long(para, chunk_size))

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if not current:
            current = block
            continue
        if len(current) + 2 + len(block) <= chunk_size:
            current = f"{current}\n\n{block}"
            continue
        chunks.append(current)
        carry = _overlap_tail(current, chunk_overlap)
        current = f"{carry}\n\n{block}".strip() if carry else block

    if current:
        chunks.append(current)
    return chunks


def stable_chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Content-addressed id: re-ingesting unchanged text overwrites in place
    rather than duplicating, which keeps `count()` honest across re-runs."""
    digest = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def chunk_document(doc: Document, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk one Document, carrying its provenance onto every Chunk."""
    out: list[Chunk] = []
    for ordinal, piece in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap)):
        out.append(
            Chunk(
                chunk_id=stable_chunk_id(doc.doc_id, ordinal, piece),
                doc_id=doc.doc_id,
                text=piece,
                ordinal=ordinal,
                source=doc.source,
                title=doc.title,
                metadata=dict(doc.metadata),
            )
        )
    return out


def chunk_documents(
    docs: Iterable[Document], chunk_size: int = 900, chunk_overlap: int = 150
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return out
