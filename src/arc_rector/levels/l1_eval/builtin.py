"""L1 offline evaluator: deterministic, no LLM judge, no network.

Ragas is the default because LLM-judged metrics capture things string overlap
cannot. But an LLM judge is slow, non-deterministic, and needs a model running --
so the eval harness would be untestable if that were the only option.

The four metrics here are computed from token overlap and citation structure:

  * `context_recall`     -- how much of the reference answer's content appears in
                            the retrieved context. Low means retrieval missed.
  * `answer_correctness`  -- token F1 between the answer and the reference.
  * `faithfulness_proxy`  -- share of answer tokens that appear in the retrieved
                            context. Low means the model wrote things the context
                            never said, which is the shape of a hallucination.
  * `citation_rate`       -- share of answers carrying at least one [n] marker.

Call them what they are: proxies. `faithfulness_proxy` cannot tell a correct
paraphrase from an invented claim that happens to reuse context vocabulary. They
are here to make the harness runnable and regression-detectable offline, not to
replace a judged metric.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ...interfaces import Evaluator

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with".split()
)
_MARKER = re.compile(r"\[\d{1,2}\]")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def _recall(reference: str, haystack: str) -> float:
    ref = _tokens(reference)
    if not ref:
        return 0.0
    return len(ref & _tokens(haystack)) / len(ref)


def _f1(answer: str, reference: str) -> float:
    a, r = _tokens(answer), _tokens(reference)
    if not a or not r:
        return 0.0
    overlap = len(a & r)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


class BuiltinEvaluator(Evaluator):
    name = "builtin"

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        per_sample: list[dict[str, Any]] = []
        for sample in samples:
            question = str(sample.get("user_input", ""))
            answer = str(sample.get("response", ""))
            reference = str(sample.get("reference", ""))
            contexts = sample.get("retrieved_contexts") or []
            context_blob = "\n".join(str(c) for c in contexts)

            per_sample.append(
                {
                    "user_input": question,
                    "context_recall": round(_recall(reference, context_blob), 4),
                    "answer_correctness": round(_f1(answer, reference), 4),
                    "faithfulness_proxy": round(_recall(answer, context_blob), 4),
                    "citation_rate": 1.0 if _MARKER.search(answer) else 0.0,
                }
            )

        keys = ["context_recall", "answer_correctness", "faithfulness_proxy", "citation_rate"]
        metrics = {
            key: round(sum(row[key] for row in per_sample) / len(per_sample), 4)
            for key in keys
        } if per_sample else {key: 0.0 for key in keys}

        return {"metrics": metrics, "per_sample": per_sample, "backend": self.name}
