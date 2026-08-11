"""A small Ragas eval harness over hand-written Q/A pairs.

The pairs below have their reference answers written from the demo corpus, so
the harness measures the system rather than the corpus. Eight pairs is not a
benchmark -- it is a regression check. Its job is to tell you whether the change
you just made to chunking, or to the embedding model, or to top_k, made things
better or worse, which is a question no amount of eyeballing one answer can
settle.

    python -m arc_rector.eval_harness
    arc-rector eval --limit 3
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .config import Config, load_config

RULE = "=" * 78


@dataclass
class QAPair:
    question: str
    reference: str


GOLDEN: list[QAPair] = [
    QAPair(
        "Which vector database is the default in Arc Rector, and why?",
        "Qdrant is the default because it runs as a single container of about 275 megabytes "
        "with no external dependencies, supports cosine distance natively, and its payload "
        "model stores the whole text chunk with the vector so retrieval needs one round trip.",
    ),
    QAPair(
        "What is the difference between open weights and open source?",
        "Open source permits use for any purpose without additional restriction, as with MIT "
        "or Apache 2.0. Open weights means only that the trained parameters are downloadable, "
        "and the licence may still restrict use. Llama 3.x uses the Llama Community Licence, "
        "which has an acceptable use policy and a 700 million monthly active user clause, so "
        "it is open weights but not open source.",
    ),
    QAPair(
        "What prefixes does Nomic Embed require and what happens if you omit them?",
        "Nomic Embed v1 and v1.5 are instruction-prefixed. Documents must be embedded with "
        "search_document: and queries with search_query:. Omitting the prefixes raises no "
        "error but measurably degrades retrieval quality, making it a silent bug.",
    ),
    QAPair(
        "Why does Qdrant reject an arbitrary hexadecimal string as a point ID?",
        "Qdrant point IDs must be either an unsigned integer or a UUID. Systems using content "
        "hashes as chunk identifiers must reinterpret those hashes as UUIDs before writing.",
    ),
    QAPair(
        "What is the maximum indexed dimensionality for pgvector?",
        "The indexed pgvector type supports at most 2000 dimensions for both HNSW and IVFFlat. "
        "Above that, search still works but falls back to a sequential scan.",
    ),
    QAPair(
        "How does two-stage retrieval with a reranker work, and why is it used?",
        "A fast bi-encoder retrieves a wide candidate set, perhaps twelve chunks, then a slower "
        "cross-encoder that sees the query and document together reranks them down to the best "
        "four. This gives most of the accuracy of a cross-encoder at a fraction of the cost, "
        "because the expensive model only ever sees a handful of documents.",
    ),
    QAPair(
        "Which models are genuinely open source rather than merely open weights?",
        "Mistral has published models under Apache 2.0 including Mistral 7B, Qwen has released "
        "several sizes under Apache 2.0, DeepSeek has published models under MIT, and Nomic "
        "Embed is Apache 2.0.",
    ),
    QAPair(
        "What is Arc Rector explicitly not?",
        "It is a complete and correct starting point, not a production system. It has no "
        "authentication, no multi-tenancy, no rate limiting, no backup strategy and no circuit "
        "breakers, all of which must be addressed before real traffic.",
    ),
]


def collect_samples(config: Config, pairs: Sequence[QAPair]) -> list[dict[str, Any]]:
    """Run each question through the live stack and gather eval samples."""
    embeddings = config.embeddings()
    store = config.vectorstore(dim=embeddings.dim)
    deps = config.deps(embeddings=embeddings, store=store)
    framework = config.framework()

    samples: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"  [{index}/{len(pairs)}] {pair.question[:66]}")
        answer = framework.run(pair.question, deps)
        samples.append(
            {
                "user_input": pair.question,
                "response": answer.text,
                "retrieved_contexts": [r.chunk.text for r in answer.retrieved],
                "reference": pair.reference,
            }
        )
    deps.tracer.flush()
    return samples


def run_eval(config: Config | None = None, limit: int = 0) -> dict[str, Any]:
    config = config or load_config()
    pairs = GOLDEN[:limit] if limit else GOLDEN

    print(f"Running {len(pairs)} question(s) through the live stack:\n")
    samples = collect_samples(config, pairs)

    print("\nScoring...")
    evaluator = config.evaluator()
    return evaluator.evaluate(samples)


def main() -> int:
    config = load_config()
    from .cli import print_stack

    print(RULE)
    print("Arc Rector -- evaluation harness")
    print(RULE)
    print_stack(config)

    result = run_eval(config)

    print(f"\n{RULE}\nScores (backend: {result['backend']})\n{RULE}")
    for metric, score in result["metrics"].items():
        rendered = "n/a" if score is None else f"{score:.4f}"
        print(f"  {metric:<30} {rendered}")
    if result.get("ragas_error"):
        print(f"\n  note: Ragas was unavailable -- {result['ragas_error']}")

    out = config.root / ".arc_rector" / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
