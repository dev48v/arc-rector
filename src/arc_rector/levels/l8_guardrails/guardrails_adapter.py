"""L8 default: Guardrails AI (Apache-2.0).

Guardrails AI is the default because it gives validation a real structure: a
`Guard` composes validators, each validator returns pass/fail with a reason, and
`on_fail` policy (exception, filter, reask, fix) is declarative rather than a
pile of if-statements.

One decision worth explaining, because it is the interesting part of this file:
**the validators here are defined locally, not installed from Guardrails Hub.**
The Hub is where the good pre-built validators live (DetectPII, ToxicLanguage,
RestrictToTopic), but `guardrails hub install` requires `guardrails configure`
with a Hub token, and this project promises to run with no keys at all. So this
adapter uses Guardrails AI as the validation *engine* and registers its own
validators through the public `@register_validator` API -- no token, no network,
same guarantees.

If you do want the Hub validators, that is a one-line swap documented in the
README: run `guardrails configure`, `guardrails hub install hub://guardrails/...`,
and add them to `_build_guards`. You are then no longer key-free, which is
exactly the kind of trade this repo tries to make visible rather than hide.

Falls back to `BuiltinGuardrails` when the package is absent and
`fallback_to_builtin` is on, so the default path never hard-fails on an optional
dependency.
"""

from __future__ import annotations

from typing import Any

from ...interfaces import Guardrails
from ...registry import require
from ...types import GuardResult
from .builtin import INJECTION_PATTERNS, SECRET_PATTERNS, BuiltinGuardrails


class GuardrailsAI(Guardrails):
    name = "guardrails-ai"

    def __init__(
        self,
        *,
        max_input_chars: int = 2000,
        min_input_chars: int = 2,
        fallback_to_builtin: bool = True,
        **_: Any,
    ) -> None:
        self.max_input_chars = max_input_chars
        self.min_input_chars = min_input_chars
        self.fallback_to_builtin = fallback_to_builtin
        self._builtin = BuiltinGuardrails(
            max_input_chars=max_input_chars, min_input_chars=min_input_chars
        )
        self._input_guard: Any = None
        self._output_guard: Any = None
        self._failed = False
        self.active_backend = "guardrails-ai"

    # -- guard construction -------------------------------------------------
    def _build_guards(self) -> tuple[Any, Any]:
        """Register local validators and compose them into input/output Guards."""
        import re

        gr = require("guardrails-ai", "guardrails-ai", "guardrails")
        base = require("guardrails-ai", "guardrails-ai", "guardrails.validator_base")

        Validator = base.Validator
        register_validator = base.register_validator
        PassResult = base.PassResult
        FailResult = base.FailResult
        OnFailAction = gr.OnFailAction

        max_chars = self.max_input_chars
        min_chars = self.min_input_chars
        injection = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        secrets = [(re.compile(p), label) for p, label in SECRET_PATTERNS]

        @register_validator(name="arc/input-length", data_type="string")
        class InputLength(Validator):  # type: ignore[misc]
            """Reject empty questions and denial-of-wallet sized ones."""

            def validate(self, value: Any, metadata: dict) -> Any:
                text = str(value or "").strip()
                if len(text) < min_chars:
                    return FailResult(error_message="Input is empty or too short to be a question.")
                if len(text) > max_chars:
                    return FailResult(
                        error_message=(
                            f"Input is {len(text)} characters, over the {max_chars}-character limit."
                        )
                    )
                return PassResult()

        @register_validator(name="arc/prompt-injection", data_type="string")
        class PromptInjection(Validator):  # type: ignore[misc]
            """Pattern-match the common instruction-override phrasings."""

            def validate(self, value: Any, metadata: dict) -> Any:
                text = str(value or "")
                for pattern in injection:
                    match = pattern.search(text)
                    if match:
                        return FailResult(
                            error_message=(
                                f"Input matches a prompt-injection pattern: {match.group(0)!r}"
                            )
                        )
                return PassResult()

        @register_validator(name="arc/no-secrets", data_type="string")
        class NoSecrets(Validator):  # type: ignore[misc]
            """Stop an answer that echoes something shaped like a credential."""

            def validate(self, value: Any, metadata: dict) -> Any:
                text = str(value or "")
                for pattern, label in secrets:
                    if pattern.search(text):
                        return FailResult(
                            error_message=f"Output appears to contain a {label}; refusing to return it."
                        )
                return PassResult()

        # Guard.use is variadic (`*validator_spread`); the older `use_many`
        # spelling was removed, so composing through `use` is the portable form.
        input_guard = gr.Guard().use(
            InputLength(on_fail=OnFailAction.EXCEPTION),
            PromptInjection(on_fail=OnFailAction.EXCEPTION),
        )
        output_guard = gr.Guard().use(NoSecrets(on_fail=OnFailAction.EXCEPTION))
        return input_guard, output_guard

    def _guards(self) -> tuple[Any, Any] | None:
        if self._failed:
            return None
        if self._input_guard is None:
            try:
                self._input_guard, self._output_guard = self._build_guards()
            except Exception:
                self._failed = True
                self.active_backend = "builtin (guardrails-ai unavailable)"
                return None
        return self._input_guard, self._output_guard

    # -- interface ----------------------------------------------------------
    def check_input(self, text: str) -> GuardResult:
        guards = self._guards()
        if guards is None:
            if not self.fallback_to_builtin:
                raise RuntimeError("guardrails-ai unavailable and fallback_to_builtin is disabled")
            return self._builtin.check_input(text)

        try:
            guards[0].validate(text or "")
        except Exception as exc:
            return GuardResult(
                allowed=False, reason=_first_reason(exc), validator="guardrails-ai/input"
            )
        return GuardResult(allowed=True, text=(text or "").strip(), validator="guardrails-ai")

    def check_output(self, text: str, context: str = "") -> GuardResult:
        guards = self._guards()
        if guards is None:
            if not self.fallback_to_builtin:
                raise RuntimeError("guardrails-ai unavailable and fallback_to_builtin is disabled")
            return self._builtin.check_output(text, context)

        try:
            guards[1].validate(text or "")
        except Exception as exc:
            return GuardResult(
                allowed=False, reason=_first_reason(exc), validator="guardrails-ai/output"
            )
        return GuardResult(allowed=True, text=text or "", validator="guardrails-ai")


def _first_reason(exc: Exception) -> str:
    """Pull the validator's message out of Guardrails' wrapper exception."""
    message = str(exc)
    marker = "Error Message: "
    if marker in message:
        return message.split(marker, 1)[1].strip().splitlines()[0]
    return message.strip().splitlines()[0] if message.strip() else exc.__class__.__name__
