"""Shared fixtures. Every test here runs with no network, no container, no model.

That is a hard rule: a test that needs Ollama or Docker is a test that will be
skipped, and a skipped test protects nothing. The offline adapters (`hash`,
`memory`, `echo`, `builtin`, `none`) exist precisely so the deterministic parts
of the stack can be tested for real.
"""

from __future__ import annotations

import pytest

from arc_rector.interfaces import AgentDeps
from arc_rector.levels.l0_inference.echo import EchoInference
from arc_rector.levels.l1_observability.noop import NoopTracer
from arc_rector.levels.l4_vectorstore.memory import InMemoryStore
from arc_rector.levels.l5_embeddings.hashing import HashEmbeddings
from arc_rector.levels.l5_embeddings.rerank import NoReranker
from arc_rector.levels.l7_memory.local import LocalMemory
from arc_rector.levels.l8_guardrails.builtin import BuiltinGuardrails
from arc_rector.types import Chunk, Document, Retrieved

CORPUS = [
    Document(
        doc_id="vectors.md",
        title="Vector Databases",
        source="corpus/vectors.md",
        text=(
            "Qdrant is the default vector database in Arc Rector. It is written in Rust "
            "and released under the Apache 2.0 licence.\n\n"
            "Qdrant point identifiers must be an unsigned integer or a UUID. An arbitrary "
            "hexadecimal string is rejected by the server.\n\n"
            "Chroma returns cosine distance rather than similarity, so scores must be "
            "converted by subtracting the distance from one."
        ),
    ),
    Document(
        doc_id="embeddings.md",
        title="Embeddings",
        source="corpus/embeddings.md",
        text=(
            "Nomic Embed is an Apache 2.0 licensed embedding model producing 768 "
            "dimensional vectors.\n\n"
            "Nomic Embed requires instruction prefixes. Documents use search_document "
            "and queries use search_query. Omitting them silently degrades retrieval."
        ),
    ),
]


@pytest.fixture
def embeddings() -> HashEmbeddings:
    return HashEmbeddings(dim=256)


@pytest.fixture
def store(embeddings: HashEmbeddings) -> InMemoryStore:
    from arc_rector.chunking import chunk_documents

    chunks = chunk_documents(CORPUS, chunk_size=400, chunk_overlap=60)
    vectors = embeddings.embed_documents([c.text for c in chunks])
    store = InMemoryStore()
    store.ensure_collection(embeddings.dim)
    store.upsert(chunks, vectors)
    return store


@pytest.fixture
def memory(tmp_path) -> LocalMemory:
    return LocalMemory(path=str(tmp_path / "memory"))


@pytest.fixture
def deps(embeddings, store, memory) -> AgentDeps:
    return AgentDeps(
        embeddings=embeddings,
        store=store,
        inference=EchoInference(),
        tracer=NoopTracer(record=True),
        memory=memory,
        guardrails=BuiltinGuardrails(),
        reranker=NoReranker(),
        top_k=3,
        fetch_k=6,
        user_id="test-user",
        max_context_chars=3000,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{i:032x}", doc_id="d.md", text=f"chunk {i}", ordinal=i,
              title="Doc", source="corpus/d.md")
        for i in range(3)
    ]


@pytest.fixture
def sample_hits(sample_chunks) -> list[Retrieved]:
    return [Retrieved(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(sample_chunks)]
