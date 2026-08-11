"""L1 alternative: DeepEval (Confident AI, Apache-2.0) -- eval as a test suite.

What this option buys you: metrics that behave like assertions. Ragas hands you a
dataframe of scores; DeepEval hands you `LLMTestCase` objects, thresholds, and a
pytest integration, so "answer relevancy must stay above 0.7" becomes a failing
build rather than a number someone eyeballs. Its metrics are also self-explaining
-- every score comes back with a `reason` string -- which is what you want when a
regression has to be argued about in a PR rather than merely observed.

The requirement you cannot skip: DeepEval defaults to OpenAI. Every metric here
is itself LLM-judged, and out of the box that judge is a hosted GPT model with
your key on it -- install, run, get billed. This repo is zero-key and local, so
this adapter subclasses `DeepEvalBaseLLM` and points the judge at the same local
Ollama endpoint the rest of the stack uses. The judge is passed explicitly to
every metric; nothing falls back to a hosted default.

Two consequences of that choice, both real. First, a local 8B judge is a weaker
grader than the frontier model DeepEval assumes, so treat scores as trend lines,
not absolutes. Second, DeepEval asks its judge for structured JSON, which small
models are bad at freehand -- so `generate` passes the requested schema to
Ollama's `format` parameter and gets constrained decoding instead of hoping. Also
note `async_mode` defaults to False here: concurrent requests to one local model
queue up anyway, and serial runs give far more readable failures.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...interfaces import Evaluator
from ...registry import require

# metric name -> (DeepEval class, test-case fields it cannot run without)
METRIC_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "answer_relevancy": ("AnswerRelevancyMetric", ("input", "actual_output")),
    "faithfulness": ("FaithfulnessMetric", ("input", "actual_output", "retrieval_context")),
    "contextual_precision": (
        "ContextualPrecisionMetric",
        ("input", "actual_output", "expected_output", "retrieval_context"),
    ),
    "contextual_recall": (
        "ContextualRecallMetric",
        ("input", "actual_output", "expected_output", "retrieval_context"),
    ),
}


def _judge_class(base: type) -> type:
    """Build the local-judge class once DeepEvalBaseLLM is importable."""

    class OllamaJudge(base):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *,
            model: str = "llama3.1:8b",
            base_url: str = "http://localhost:11434",
            timeout: int = 180,
            temperature: float = 0.0,
        ) -> None:
            # Set these first: the base __init__ calls load_model().
            self._model = model
            self._base_url = base_url.rstrip("/")
            self._timeout = timeout
            self._temperature = temperature
            self._requests = require("requests", "deepeval")
            super().__init__(model)

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            return self

        def get_model_name(self) -> str:
            return f"ollama/{self._model}"

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            payload: dict[str, Any] = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self._temperature},
            }
            if schema is not None:
                # Constrained decoding beats asking a small model to behave.
                payload["format"] = schema.model_json_schema()

            response = self._requests.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            text = str(response.json().get("response", "")).strip()

            if schema is None:
                return text
            return schema.model_validate_json(text)

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return self.generate(prompt, schema, **kwargs)

    return OllamaJudge


class DeepEvalEvaluator(Evaluator):
    name = "deepeval"

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        metrics: Sequence[str] = ("answer_relevancy", "faithfulness"),
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = False,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.metric_names = tuple(metrics)
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self._judge: Any = None
        self._metrics: dict[str, Any] | None = None
        self._test_case_cls: Any = None

    @property
    def judge(self) -> Any:
        """The local judge. Built on first use so import never needs DeepEval."""
        if self._judge is None:
            models = require("deepeval", self.name, import_name="deepeval.models")
            self._judge = _judge_class(models.DeepEvalBaseLLM)(
                model=self.model, base_url=self.base_url, timeout=self.timeout
            )
        return self._judge

    @property
    def test_case_cls(self) -> Any:
        if self._test_case_cls is None:
            module = require("deepeval", self.name, import_name="deepeval.test_case")
            self._test_case_cls = module.LLMTestCase
        return self._test_case_cls

    @property
    def metrics(self) -> dict[str, Any]:
        """Instantiate each configured metric, wired to the local judge."""
        if self._metrics is None:
            module = require("deepeval", self.name, import_name="deepeval.metrics")
            built: dict[str, Any] = {}
            for metric_name in self.metric_names:
                spec = METRIC_SPECS.get(metric_name)
                if spec is None:
                    raise ValueError(
                        f"Unknown DeepEval metric {metric_name!r}. "
                        f"Known: {', '.join(sorted(METRIC_SPECS))}"
                    )
                cls = getattr(module, spec[0])
                built[metric_name] = cls(
                    threshold=self.threshold,
                    model=self.judge,
                    include_reason=self.include_reason,
                    async_mode=self.async_mode,
                )
            self._metrics = built
        return self._metrics

    def evaluate(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {"metrics": {}, "per_sample": [], "backend": self.name}

        metrics = self.metrics
        test_case_cls = self.test_case_cls

        per_sample: list[dict[str, Any]] = []
        totals: dict[str, list[float]] = {name: [] for name in metrics}

        for sample in samples:
            fields = self._fields(sample)
            case = test_case_cls(**fields)
            row: dict[str, Any] = {
                "user_input": fields["input"],
                "response": fields["actual_output"],
                "scores": {},
                "reasons": {},
            }

            for metric_name, metric in metrics.items():
                missing = self._missing(metric_name, fields)
                if missing:
                    row["reasons"][metric_name] = f"skipped: no {', '.join(missing)}"
                    continue
                try:
                    metric.measure(case)
                except Exception as exc:
                    row["reasons"][metric_name] = f"error: {type(exc).__name__}: {exc}"
                    continue

                score = metric.score
                if score is None:
                    row["reasons"][metric_name] = "error: metric returned no score"
                    continue
                row["scores"][metric_name] = float(score)
                totals[metric_name].append(float(score))
                if self.include_reason and getattr(metric, "reason", None):
                    row["reasons"][metric_name] = str(metric.reason)

            per_sample.append(row)

        averages = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in totals.items()
            if scores
        }
        return {"metrics": averages, "per_sample": per_sample, "backend": self.name}

    @staticmethod
    def _fields(sample: dict[str, Any]) -> dict[str, Any]:
        """Map Arc Rector's sample keys onto LLMTestCase's."""
        contexts = sample.get("retrieved_contexts") or []
        return {
            "input": str(sample.get("user_input", "")),
            "actual_output": str(sample.get("response", "")),
            "expected_output": str(sample.get("reference", "")) or None,
            "retrieval_context": [str(c) for c in contexts] or None,
        }

    @staticmethod
    def _missing(metric_name: str, fields: dict[str, Any]) -> list[str]:
        """Which required test-case fields this sample does not supply."""
        required = METRIC_SPECS[metric_name][1]
        return [key for key in required if not fields.get(key)]
