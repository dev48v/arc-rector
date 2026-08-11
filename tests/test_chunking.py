"""Chunking is pure and deterministic, so it gets tested properly."""

from __future__ import annotations

import pytest

from arc_rector.chunking import (
    chunk_document,
    chunk_text,
    normalise,
    stable_chunk_id,
)
from arc_rector.types import Document


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("A single short paragraph.", chunk_size=500)
    assert chunks == ["A single short paragraph."]


def test_every_chunk_respects_the_size_limit():
    text = "\n\n".join(f"Paragraph number {i} with some filler words in it." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # Overlap is prepended, so allow the carry on top of the nominal size.
    assert all(len(c) <= 200 + 40 + 2 for c in chunks), [len(c) for c in chunks]


def test_paragraphs_are_packed_not_split_when_they_fit():
    text = "First para.\n\nSecond para.\n\nThird para."
    assert chunk_text(text, chunk_size=500) == [text]


def test_oversized_paragraph_is_split_on_sentences():
    sentence = "This is a sentence of a reasonably predictable length. "
    chunks = chunk_text(sentence * 20, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_a_single_sentence_longer_than_the_limit_is_hard_split():
    chunks = chunk_text("x" * 500, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 5
    assert all(len(c) == 100 for c in chunks)


def test_overlap_carries_context_between_chunks():
    paras = "\n\n".join(f"Paragraph {i} carries distinctive token{i}." for i in range(12))
    with_overlap = chunk_text(paras, chunk_size=150, chunk_overlap=60)
    without = chunk_text(paras, chunk_size=150, chunk_overlap=0)
    # Overlap duplicates content, so the same text yields at least as many chunks.
    assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without)


def test_no_content_is_lost():
    text = "\n\n".join(f"Unique marker {i} here." for i in range(15))
    joined = " ".join(chunk_text(text, chunk_size=120, chunk_overlap=30))
    for i in range(15):
        assert f"Unique marker {i}" in joined


def test_chunking_is_deterministic():
    text = "\n\n".join(f"Para {i} content." for i in range(25))
    assert chunk_text(text, 200, 50) == chunk_text(text, 200, 50)


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_normalise_collapses_newlines_and_strips():
    assert normalise("a\r\n\r\n\r\n\r\nb   \n") == "a\n\nb"


def test_chunk_ids_are_content_addressed():
    assert stable_chunk_id("d", 0, "same") == stable_chunk_id("d", 0, "same")
    assert stable_chunk_id("d", 0, "a") != stable_chunk_id("d", 0, "b")
    assert stable_chunk_id("d", 0, "a") != stable_chunk_id("d", 1, "a")
    assert len(stable_chunk_id("d", 0, "a")) == 32


def test_chunk_document_carries_provenance():
    doc = Document(
        doc_id="paper.pdf", text="\n\n".join(f"Para {i}." for i in range(10)),
        source="corpus/paper.pdf", title="A Paper", metadata={"lang": "en"},
    )
    chunks = chunk_document(doc, chunk_size=60, chunk_overlap=10)
    assert len(chunks) > 1
    for ordinal, chunk in enumerate(chunks):
        assert chunk.doc_id == "paper.pdf"
        assert chunk.title == "A Paper"
        assert chunk.source == "corpus/paper.pdf"
        assert chunk.ordinal == ordinal
        assert chunk.metadata["lang"] == "en"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_payload_round_trip_preserves_the_chunk():
    from arc_rector.types import Chunk

    original = Chunk(
        chunk_id="a" * 32, doc_id="d.md", text="body", ordinal=2,
        source="corpus/d.md", title="T", metadata={"extra": "kept"},
    )
    restored = Chunk.from_payload(original.payload())
    assert restored.chunk_id == original.chunk_id
    assert restored.text == original.text
    assert restored.ordinal == original.ordinal
    assert restored.title == original.title
    assert restored.metadata["extra"] == "kept"
