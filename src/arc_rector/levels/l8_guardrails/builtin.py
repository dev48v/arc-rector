"""L8 offline option: dependency-free guardrails, and the honest baseline.

Guardrails AI is the default, but this module exists for two reasons: the test
suite must be able to prove a rejection without installing anything, and every
guardrail conversation should start from "what does a regex actually catch?"

What this catches:
  * input longer than a configured limit (a cheap denial-of-wallet guard)
  * empty or whitespace-only input
  * prompt-injection phrasings from a small pattern list
  * output containing something shaped like a secret (API keys, private keys)
  * output that is ungrounded: it cites nothing when context was supplied

What this does NOT catch: anything paraphrased around the patterns. Regex
guardrails are a speed bump, not a wall. Treat this as defence in depth beneath
a model-based check (see the `llamaguard` and `nemo` adapters), never as your
only layer -- the README says the same thing in plainer words.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Pattern

from ...citations import used_markers
from ...interfaces import Guardrails
from ...types import GuardResult

# Qualifiers stack in real attacks ("disregard your prior rules"), so the
# qualifier group repeats rather than matching exactly once.
_QUAL = r"(?:(?:all|any|the|your|my|these|those|previous|prior|above|earlier|preceding)\s+)*"

INJECTION_PATTERNS: tuple[str, ...] = (
    rf"ignore\s+{_QUAL}(?:instructions?|prompts?|rules?|directions?)",
    rf"disregard\s+{_QUAL}(?:instructions?|rules?|prompts?|directions?)",
    r"forget\s+(everything|all)\s+(you|above|previous)",
    r"you\s+are\s+now\s+(a|an|in)\s+\w+\s*(mode|assistant|model)?",
    r"(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|initial\s+prompt)",
    r"\bdeveloper\s+mode\b",
    r"\bDAN\b\s+mode",
    r"pretend\s+(you\s+are|to\s+be)\s+(not\s+)?(an?\s+)?(unrestricted|jailbroken|evil)",
    r"</?(system|instruction)>",
)

SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsk-[A-Za-z0-9]{20,}\b", "OpenAI-style secret key"),
    (r"\bnvapi-[A-Za-z0-9_\-]{20,}\b", "NVIDIA API key"),
    (r"\bghp_[A-Za-z0-9]{30,}\b", "GitHub personal access token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "Slack token"),
)


def _compile(patterns: Iterable[str]) -> list[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class BuiltinGuardrails(Guardrails):
    name = "builtin"

    def __init__(
        self,
        *,
        max_input_chars: int = 2000,
        min_input_chars: int = 2,
        block_injection: bool = True,
        redact_secrets: bool = True,
        require_citations: bool = False,
        **_: Any,
    ) -> None:
        self.max_input_chars = max_input_chars
        self.min_input_chars = min_input_chars
        self.block_injection = block_injection
        self.redact_secrets = redact_secrets
        self.require_citations = require_citations
        self._injection = _compile(INJECTION_PATTERNS)
        self._secrets = [(re.compile(p), label) for p, label in SECRET_PATTERNS]

    def check_input(self, text: str) -> GuardResult:
        stripped = (text or "").strip()

        if len(stripped) < self.min_input_chars:
            return GuardResult(
                allowed=False,
                reason="Input is empty or too short to be a question.",
                validator="min-length",
            )

        if len(stripped) > self.max_input_chars:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Input is {len(stripped)} characters, over the "
                    f"{self.max_input_chars}-character limit."
                ),
                validator="max-length",
            )

        if self.block_injection:
            for pattern in self._injection:
                match = pattern.search(stripped)
                if match:
                    return GuardResult(
                        allowed=False,
                        reason=f"Input matches a prompt-injection pattern: {match.group(0)!r}",
                        validator="prompt-injection",
                    )

        return GuardResult(allowed=True, text=stripped, validator="builtin")

    def check_context(self, context: str) -> GuardResult:
        """Indirect injection: the attack is in the document, never in the question.

        This reports rather than rejects. A corpus legitimately containing the
        phrase "ignore all previous instructions" -- a security document, this
        project's own README -- must still be answerable, so the caller turns a
        hit into an explicit warning in the prompt instead of a refusal.
        """
        text = context or ""
        if not text.strip() or not self.block_injection:
            return GuardResult(allowed=True, text=text, validator=self.name)

        for pattern in self._injection:
            match = pattern.search(text)
            if match:
                return GuardResult(
                    allowed=False,
                    text=text,
                    reason=(
                        "Retrieved documents contain instruction-shaped text: "
                        f"{match.group(0)!r}"
                    ),
                    validator="indirect-injection",
                )
        return GuardResult(allowed=True, text=text, validator=self.name)

    def check_output(self, text: str, context: str = "") -> GuardResult:
        out = text or ""

        if self.redact_secrets:
            for pattern, label in self._secrets:
                if pattern.search(out):
                    return GuardResult(
                        allowed=False,
                        reason=f"Output appears to contain a {label}; refusing to return it.",
                        validator="secret-leak",
                    )

        if self.require_citations and context.strip():
            if not used_markers(out) and "don't know" not in out.lower():
                return GuardResult(
                    allowed=False,
                    reason="Answer cited no sources despite context being supplied.",
                    validator="citation-required",
                )

        return GuardResult(allowed=True, text=out, validator="builtin")


class NoGuardrails(Guardrails):
    """L8 disabled. Present so the level is explicit rather than implicit."""

    name = "none"

    def __init__(self, **_: Any) -> None:
        pass

    def check_input(self, text: str) -> GuardResult:
        return GuardResult(allowed=True, text=text, validator="none")

    def check_output(self, text: str, context: str = "") -> GuardResult:
        return GuardResult(allowed=True, text=text, validator="none")
