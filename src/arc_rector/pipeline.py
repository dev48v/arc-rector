"""The ingestion pipeline: sources -> parse (L6) -> chunk -> embed (L5) -> store (L4).

Deliberately boring and level-agnostic. It never imports a concrete adapter; it
takes whatever `Config` built. That is why re-running with
`ARC_L4_VECTORSTORE=chroma` re-ingests into Chroma with no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import chunk_documents
from .config import Config
from .interfaces import Embeddings, Loader, VectorStore
from .types import Chunk, Document

INGESTIBLE_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".pdf", ".docx", ".pptx",
    ".xlsx", ".html", ".htm", ".epub", ".csv",
}


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    written: int = 0
    stored_total: int = 0
    dim: int = 0
    store: str = ""
    embedder: str = ""
    loader: str = ""
    failures: list[tuple[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"  parsed     {self.documents} documents with {self.loader}",
            f"  chunked    {self.chunks} chunks",
            f"  embedded   {self.written} vectors, dim {self.dim}, via {self.embedder}",
            f"  stored     {self.stored_total} points total in {self.store}",
        ]
        if self.failures:
            lines.append(f"  failed     {len(self.failures)} source(s):")
            lines.extend(f"               {src}: {err}" for src, err in self.failures)
        return "\n".join(lines)


def discover(corpus_dir: Path) -> list[Path]:
    """Every ingestible file under `corpus_dir`, sorted for reproducibility."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    return sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in INGESTIBLE_SUFFIXES
    )


def load_documents(loader: Loader, sources: Iterable[str | Path]) -> tuple[list[Document], list[tuple[str, str]]]:
    """Parse each source, collecting failures rather than aborting the batch."""
    docs: list[Document] = []
    failures: list[tuple[str, str]] = []
    for source in sources:
        text_source = str(source)
        try:
            doc = loader.load(text_source)
            if doc.text.strip():
                docs.append(doc)
            else:
                failures.append((text_source, "parsed to empty text"))
        except Exception as exc:
            failures.append((text_source, f"{exc.__class__.__name__}: {exc}"))
    return docs, failures


def embed_and_store(
    chunks: Sequence[Chunk],
    embeddings: Embeddings,
    store: VectorStore,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Embed in batches and upsert. Returns (written, vector_dim)."""
    if not chunks:
        return 0, embeddings.dim

    first = embeddings.embed_documents([chunks[0].text])
    dim = len(first[0])
    store.ensure_collection(dim)

    written = store.upsert(chunks[:1], first)
    for start in range(1, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        vectors = embeddings.embed_documents([c.text for c in batch])
        written += store.upsert(batch, vectors)
    return written, dim


def ingest(
    config: Config,
    sources: Sequence[str | Path] | None = None,
    *,
    reset: bool = False,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
    loader: Loader | None = None,
    batch_size: int = 32,
) -> IngestReport:
    """Run the full ingestion pipeline with whatever levels config selected."""
    loader = loader or config.loader()
    embeddings = embeddings or config.embeddings()
    store = store or config.vectorstore(dim=embeddings.dim)

    paths: Sequence[str | Path] = sources if sources else discover(config.corpus_dir())
    docs, failures = load_documents(loader, paths)

    pipe = config.pipeline
    chunks = chunk_documents(docs, int(pipe["chunk_size"]), int(pipe["chunk_overlap"]))

    if reset:
        # Drop before writing so a dimension change from an L5 swap cannot
        # collide with the existing collection.
        store.drop()

    written, dim = embed_and_store(chunks, embeddings, store, batch_size)

    return IngestReport(
        documents=len(docs),
        chunks=len(chunks),
        written=written,
        stored_total=store.count(),
        dim=dim,
        store=f"{config.use('l4_vectorstore')}",
        embedder=f"{config.use('l5_embeddings')}",
        loader=f"{config.use('l6_ingestion')}",
        failures=failures,
        sources=[str(p) for p in paths],
    )
