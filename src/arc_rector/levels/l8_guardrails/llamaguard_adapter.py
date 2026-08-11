"""L8 alternative: Meta's Llama Guard 3, served locally by Ollama.

What this option buys you: a safety classifier that is a *model*, so it reads
meaning rather than patterns. The builtin regex guard is trivially paraphrased
around; Llama Guard was fine-tuned on the MLCommons hazard taxonomy and answers
with a category, which is why this adapter can tell you a message was blocked for
S9 Indiscriminate Weapons rather than "matched a pattern". It is also the cheapest
model-based option here: no rail LLM to prompt like NeMo, no gated download and no
HuggingFace token like LlamaFirewall -- `ollama pull llama-guard3:1b` and it runs
on the same endpoint the rest of the stack already uses.

LICENCE, stated plainly: Llama Guard 3 is OPEN WEIGHTS under the Llama 3.2
Community Licence -- NOT OSI open source. You may run and fine-tune it, but the
licence carries Meta's Acceptable Use Policy, an attribution requirement, and a
monthly-active-user threshold above which you must ask Meta for a separate
licence. Everything else in this repo is Apache-2.0 or MIT; this one is not, and
that is the tradeoff you accept for the best small classifier available.

The gotcha: the 1b and 8b variants do not share a taxonomy. Llama Guard 3-8B
covers S1..S14, where S14 is Code Interpreter Abuse; the 1B model was trained on
S1..S13 only and will never emit S14. Do not write logic that assumes a category
exists on both.

Fail-closed by design: if Ollama is unreachable or the model is not pulled, both
checks return not-allowed with the reason rather than waving text through. Pass
`fail_open=True` to reverse that.
"""

from __future__ import annotations

from typing import Any

from ...interfaces import Guardrails
from ...registry import require
from ...types import GuardResult

# MLCommons hazard taxonomy as used by Llama Guard 3. S14 is 8B-only.
HAZARDS: dict[str, str] = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


def hazard_name(code: str) -> str:
    """Human-readable name for an S-code, or the code itself if unrecognised."""
    return HAZARDS.get(code.strip().upper(), f"Unmapped category {code.strip().upper()}")


class LlamaGuard(Guardrails):
    name = "llamaguard"

    def __init__(
        self,
        *,
        model: str = "llama-guard3:1b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_input_chars: int = 2000,
        fail_open: bool = False,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.fail_open = fail_open
        self._requests: Any = None

    @property
    def requests(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", self.name)
        return self._requests

    def check_input(self, text: str) -> GuardResult:
        stripped = (text or "").strip()

        if not stripped:
            return GuardResult(
                allowed=False, reason="Input is empty.", validator="llamaguard:min-length"
            )

        if len(stripped) > self.max_input_chars:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Input is {len(stripped)} characters, over the "
                    f"{self.max_input_chars}-character limit."
                ),
                validator="llamaguard:max-length",
            )

        messages = [{"role": "user", "content": stripped}]
        return self._classify(messages, stripped, "llamaguard:input")

    def check_output(self, text: str, context: str = "") -> GuardResult:
        out = text or ""

        # Llama Guard grades the last turn, so the user turn must precede it.
        messages = [
            {"role": "user", "content": context.strip() or "(context not supplied)"},
            {"role": "assistant", "content": out},
        ]
        return self._classify(messages, out, "llamaguard:output")

    def _classify(
        self, messages: list[dict[str, str]], text: str, validator: str
    ) -> GuardResult:
        try:
            verdict = self._chat(messages)
        except Exception as exc:
            return self._unavailable(text, exc, validator)
        return self._parse(verdict, text, validator)

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """One /api/chat call. Ollama applies the Llama Guard prompt template."""
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return str((body.get("message") or {}).get("content", "")).strip()

    def _parse(self, verdict: str, text: str, validator: str) -> GuardResult:
        """Llama Guard answers "safe", or "unsafe" then a line of S-codes."""
        lines = [line.strip() for line in verdict.splitlines() if line.strip()]
        if not lines:
            return self._unavailable(text, ValueError("empty classifier response"), validator)

        head = lines[0].lower()

        if head.startswith("safe"):
            return GuardResult(allowed=True, text=text, reason="safe", validator=validator)

        if not head.startswith("unsafe"):
            # An off-format answer is not a pass. Treat it as a failed check.
            return self._unavailable(
                text, ValueError(f"unrecognised verdict {verdict!r}"), validator
            )

        codes = [c.strip().upper() for c in ",".join(lines[1:]).split(",") if c.strip()]
        if not codes:
            return GuardResult(
                allowed=False,
                reason="Llama Guard flagged this as unsafe (no category returned).",
                validator=validator,
            )

        named = ", ".join(f"{code} {hazard_name(code)}" for code in codes)
        return GuardResult(
            allowed=False,
            reason=f"Llama Guard flagged this as unsafe: {named}.",
            validator=f"{validator}:{codes[0]}",
        )

    def _unavailable(self, text: str, exc: Exception, validator: str) -> GuardResult:
        """Classifier could not run. Fail closed unless explicitly told otherwise."""
        reason = (
            f"Llama Guard could not classify ({type(exc).__name__}: {exc}). "
            f"Check Ollama at {self.base_url} and `ollama pull {self.model}`."
        )
        if self.fail_open:
            return GuardResult(
                allowed=True, text=text, reason=f"{reason} fail_open=True.", validator=validator
            )
        return GuardResult(allowed=False, reason=reason, validator=validator)
