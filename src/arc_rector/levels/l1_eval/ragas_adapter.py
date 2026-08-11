"""L1 default evaluator: Ragas (Apache-2.0).

Ragas scores a RAG system on the axes that actually matter and that string
overlap cannot see:

  * **faithfulness**    -- is every claim in the answer supported by the retrieved
                           context? This is the hallucination metric.
  * **answer_relevancy** -- does the answer address the question asked?
  * **context_precision / context_recall** -- did retrieval fetch the right
                           chunks, and enough of them? Separating these from the
                           answer metrics is the point: it tells you whether a bad
                           answer is a retrieval bug or a generation bug.

Zero-key catch: Ragas defaults to OpenAI for its judge model and its embeddings.
This adapter wires it to the same local Ollama models the rest of the stack uses,
via LangChain wrappers, so no key is needed. That means the judge is a small
local model -- treat the absolute numbers as directional and the *deltas between
runs* as the real signal.

If Ragas or its judge is unavailable, falls back to `BuiltinEvaluator` (when
`fallback_to_builtin` is on) so `make eval` always produces numbers.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...interfaces import Evaluator
from ...registry import require
from .builtin import BuiltinEvaluator

DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RagasEvaluator(Evaluator):
    name = "ragas"

    def __init__(
        self,
        *,
        llm_model: str = "llama3.2:3b",
        embed_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        metrics: Sequence[str] = DEFAULT_METRICS,
        fallback_to_builtin: bool = True,
        timeout: int = 300,
        **_: Any,
    ) -> None:
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.ollama_url = ollama_url
        self.metric_names = list(metrics)
        self.fallback_to_builtin = fallback_to_builtin
        self.timeout = timeout
        self.active_backend = "ragas"

    def _judge(self) -> tuple[Any, Any]:
        """Wrap local Ollama as the Ragas judge LLM and embedder."""
        ollama = require("langchain-ollama", "ragas", "langchain_ollama")
        wrappers = require("ragas", "ragas", "ragas.llms")
        embed_wrappers = require("ragas", "ragas", "ragas.embeddings")

        chat = ollama.ChatOllama(model=self.llm_model, base_url=self.ollama_url, temperature=0.0)
        embedder = ollama.OllamaEmbeddings(model=self.embed_model, base_url=self.ollama_url)
        return (
            wrappers.LangchainLLMWrapper(chat),
            embed_wrappers.LangchainEmbeddingsWrapper(embedder),
        )

    def _collect_metrics(self, llm: Any, embeddings: Any) -> list[Any]:
        metrics_mod = require("ragas", "ragas", "ragas.metrics")
        chosen: list[Any] = []
        for metric_name in self.metric_names:
            metric = getattr(metrics_mod, metric_name, None)
            if metric is None:
                continue
            # Ragas metrics carry their own llm/embeddings handles; setting them
            # here is what keeps the judge local instead of reaching for OpenAI.
            if hasattr(metric, "llm"):
                metric.llm = llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings
            chosen.append(metric)
        return chosen

    def evaluate(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        try:
            return self._evaluate_with_ragas(samples)
        except Exception as exc:
            if not self.fallback_to_builtin:
                raise
            result = BuiltinEvaluator().evaluate(samples)
            result["backend"] = "builtin (ragas unavailable)"
            result["ragas_error"] = str(exc)[:300]
            self.active_backend = result["backend"]
            return result

    def _evaluate_with_ragas(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ragas = require("ragas", "ragas", "ragas")
        dataset_mod = require("ragas", "ragas", "ragas.dataset_schema")

        llm, embeddings = self._judge()
        metrics = self._collect_metrics(llm, embeddings)
        if not metrics:
            raise RuntimeError(f"None of the requested Ragas metrics exist: {self.metric_names}")

        rows = [
            dataset_mod.SingleTurnSample(
                user_input=str(s.get("user_input", "")),
                response=str(s.get("response", "")),
                retrieved_contexts=[str(c) for c in (s.get("retrieved_contexts") or [])],
                reference=str(s.get("reference", "")),
            )
            for s in samples
        ]
        dataset = dataset_mod.EvaluationDataset(samples=rows)

        result = ragas.evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

        frame = result.to_pandas()
        numeric = frame.select_dtypes("number")
        metric_scores = {
            str(col): (None if numeric[col].isna().all() else round(float(numeric[col].mean()), 4))
            for col in numeric.columns
        }
        per_sample = [
            {k: (None if _is_nan(v) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        return {"metrics": metric_scores, "per_sample": per_sample, "backend": self.name}


def _is_nan(value: Any) -> bool:
    try:
        return value != value  # NaN is the only value unequal to itself.
    except Exception:
        return False
