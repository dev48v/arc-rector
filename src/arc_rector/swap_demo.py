"""Proof that the layers really are swappable: one question, four stacks.

    python -m arc_rector.swap_demo

Runs the same question across two vector databases and two agent frameworks and
prints all four answers side by side. The point is not that the answers are
identical -- a language model is not deterministic -- but that all four are
*comparably grounded*, cite real sources, and required no code change to produce.

If a swap silently degraded quality, this is where it would show.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from .config import load_config
from .pipeline import ingest

QUESTION = "Which vector database is the default in Arc Rector, and what is the reason given?"
RULE = "=" * 78

# (label, level overrides). Each combination is a complete, independent stack.
COMBINATIONS: list[tuple[str, dict[str, str]]] = [
    ("qdrant + langgraph", {"ARC_L4_VECTORSTORE": "qdrant", "ARC_L3_FRAMEWORK": "langgraph"}),
    ("qdrant + simple", {"ARC_L4_VECTORSTORE": "qdrant", "ARC_L3_FRAMEWORK": "simple"}),
    ("chroma + langgraph", {"ARC_L4_VECTORSTORE": "chroma", "ARC_L3_FRAMEWORK": "langgraph"}),
    ("chroma + simple", {"ARC_L4_VECTORSTORE": "chroma", "ARC_L3_FRAMEWORK": "simple"}),
]


def run_one(label: str, overrides: dict[str, str], question: str) -> dict[str, Any]:
    """Build a whole stack from scratch under `overrides` and answer once."""
    previous = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        config = load_config()
        embeddings = config.embeddings()
        store = config.vectorstore(dim=embeddings.dim)

        # Each vector store needs its own ingest -- that is the honest cost of a
        # swap, and hiding it would make the demo a lie.
        report = ingest(config, reset=True, embeddings=embeddings, store=store, loader=config.loader())

        deps = config.deps(embeddings=embeddings, store=store)
        answer = config.framework().run(question, deps)
        deps.tracer.flush()
        return {
            "label": label,
            "ok": not answer.blocked and bool(answer.text),
            "answer": answer.text,
            "citations": [c.render() for c in answer.citations],
            "retrieved": len(answer.retrieved),
            "vectors": report.stored_total,
            "top_score": round(answer.retrieved[0].score, 4) if answer.retrieved else None,
        }
    except Exception as exc:
        return {
            "label": label,
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    print(RULE)
    print("Arc Rector -- layer swap demo")
    print(RULE)
    print(f"\nOne question, four stacks, zero code changes.\n\nQ: {QUESTION}\n")

    results = [run_one(label, overrides, QUESTION) for label, overrides in COMBINATIONS]

    for result in results:
        print(f"\n{RULE}\n{result['label']}\n{RULE}")
        if not result.get("ok"):
            print(f"  FAILED: {result.get('error', 'unknown')}")
            continue
        print(f"  vectors stored: {result['vectors']}   retrieved: {result['retrieved']}"
              f"   top score: {result['top_score']}")
        print(f"\n{result['answer']}\n")
        for citation in result["citations"]:
            print(f"  {citation}")

    print(f"\n{RULE}\nSummary\n{RULE}")
    for result in results:
        mark = "PASS" if result.get("ok") else "FAIL"
        detail = "" if result.get("ok") else f" -- {result.get('error', '')[:80]}"
        print(f"  [{mark}] {result['label']}{detail}")

    failures = [r for r in results if not r.get("ok")]
    print()
    if failures:
        print(f"{len(failures)} of {len(results)} stacks failed.")
        return 1
    print(f"All {len(results)} stacks answered from the same corpus with no code changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
