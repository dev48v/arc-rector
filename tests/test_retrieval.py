"""Retrieval ranking, embedding determinism, and reranking."""

from __future__ import annotations

import pytest

from arc_rector.levels.l4_vectorstore.memory import InMemoryStore, cosine
from arc_rector.levels.l5_embeddings.hashing import HashEmbeddings, tokenize
from arc_rector.levels.l5_embeddings.rerank import NoReranker
from arc_rector.types import Chunk


# -- embeddings ------------------------------------------------------------
def test_embeddings_are_deterministic_across_instances():
    a = HashEmbeddings(dim=128).embed_query("Qdrant is the default")
    b = HashEmbeddings(dim=128).embed_query("Qdrant is the default")
    assert a == b


def test_embeddings_are_unit_length():
    vector = HashEmbeddings(dim=128).embed_query("some text about vectors")
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_embedding_dimension_is_respected():
    assert len(HashEmbeddings(dim=64).embed_query("text")) == 64


def test_empty_text_still_yields_a_usable_vector():
    # A zero vector makes cosine undefined downstream, so this must not happen.
    vector = HashEmbeddings(dim=32).embed_query("")
    assert any(v != 0.0 for v in vector)


def test_similar_text_scores_higher_than_unrelated_text():
    embeddings = HashEmbeddings(dim=512)
    query = embeddings.embed_query("qdrant point identifiers must be uuid")
    close = embeddings.embed_query("qdrant point identifiers must be a uuid or integer")
    far = embeddings.embed_query("nomic embed requires instruction prefixes")
    assert cosine(query, close) > cosine(query, far)


def test_tokenizer_drops_stopwords():
    assert "the" not in tokenize("the qdrant database")
    assert "qdrant" in tokenize("the qdrant database")


def test_invalid_dimension_is_rejected():
    with pytest.raises(ValueError):
        HashEmbeddings(dim=0)


# -- store ranking ---------------------------------------------------------
def test_search_returns_results_in_descending_score_order(store, embeddings):
    hits = store.search(embeddings.embed_query("qdrant point identifiers"), top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_the_right_chunk(store, embeddings):
    hits = store.search(embeddings.embed_query("Qdrant point identifiers UUID integer"), top_k=3)
    assert "identifiers" in hits[0].chunk.text.lower()


def test_top_k_is_honoured(store, embeddings):
    assert len(store.search(embeddings.embed_query("qdrant"), top_k=2)) == 2


def test_count_reflects_what_was_stored(store):
    assert store.count() > 0


def test_drop_empties_the_store(store):
    store.drop()
    assert store.count() == 0
    assert store.search([0.1] * 256, top_k=3) == []


def test_upsert_is_idempotent_for_identical_chunks(embeddings):
    store = InMemoryStore()
    store.ensure_collection(embeddings.dim)
    chunk = Chunk(chunk_id="c" * 32, doc_id="d", text="hello world", ordinal=0)
    vector = embeddings.embed_documents([chunk.text])
    store.upsert([chunk], vector)
    store.upsert([chunk], vector)
    assert store.count() == 1


def test_mismatched_chunks_and_vectors_are_rejected(embeddings):
    store = InMemoryStore()
    store.ensure_collection(embeddings.dim)
    chunk = Chunk(chunk_id="a" * 32, doc_id="d", text="t", ordinal=0)
    with pytest.raises(ValueError):
        store.upsert([chunk], [])


def test_ties_break_deterministically(embeddings):
    store = InMemoryStore()
    store.ensure_collection(4)
    vector = [1.0, 0.0, 0.0, 0.0]
    chunks = [Chunk(chunk_id=f"{i:032x}", doc_id="d", text="same", ordinal=i) for i in range(5)]
    store.upsert(chunks, [vector] * 5)
    first = [h.chunk.chunk_id for h in store.search(vector, top_k=5)]
    second = [h.chunk.chunk_id for h in store.search(vector, top_k=5)]
    assert first == second


def test_payload_survives_the_round_trip_through_the_store(store, embeddings):
    hit = store.search(embeddings.embed_query("nomic embed prefixes"), top_k=1)[0]
    assert hit.chunk.title
    assert hit.chunk.source
    assert hit.chunk.doc_id


# -- cosine ----------------------------------------------------------------
def test_cosine_handles_the_zero_vector():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_of_opposites_is_negative_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# -- reranking -------------------------------------------------------------
def test_no_reranker_preserves_order_and_scores(sample_hits):
    out = NoReranker().rerank("query", sample_hits, top_k=3)
    assert [h.chunk.chunk_id for h in out] == [h.chunk.chunk_id for h in sample_hits]
    assert [h.score for h in out] == [h.score for h in sample_hits]


def test_no_reranker_honours_top_k(sample_hits):
    assert len(NoReranker().rerank("query", sample_hits, top_k=2)) == 2


def test_no_reranker_accepts_the_cross_encoder_settings_block():
    # config.yaml carries `model:` under l5_reranker even when `use: none`.
    NoReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
