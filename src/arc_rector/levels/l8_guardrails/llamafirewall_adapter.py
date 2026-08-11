"""L8 alternative: Meta's LlamaFirewall (MIT) -- classifier-based, agent-aware.

What this option buys you: purpose-built detectors instead of prompts. PromptGuard
is a small BERT-style classifier trained on jailbreak and injection attempts, so
it runs in milliseconds on CPU and does not need a generative model in the loop
the way NeMo does. LlamaFirewall also reaches past the chat box: AlignmentCheck
audits an agent's *reasoning trace* for goal hijacking, and CodeShield statically
analyses generated code. Nothing else at this level covers the agentic surface.

The gotcha for a zero-key project like this one: the scanners are model weights,
not code. First run downloads them into `~/.cache/huggingface`, PromptGuard lives
in a gated Meta repo you must accept a licence for and authenticate against
(`hf auth login`), and AlignmentCheck needs an LLM of its own -- by default a
hosted Together endpoint, so it wants an API key. That is why this adapter
defaults to PromptGuard alone on both directions: it is the part that works with
nothing but a one-time model download. Opt into the rest with `input_scanners`
and `output_scanners`, and run `llamafirewall configure` once first.

No credential is read, logged or embedded here. The library picks its own tokens
up from the environment; this adapter never touches or echoes them.

Fail-closed by design: if the firewall cannot be built or a scan raises, both
checks return not-allowed with the reason. Pass `fail_open=True` to reverse that.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...interfaces import Guardrails
from ...registry import AdapterUnavailable, require
from ...types import GuardResult


class LlamaFirewallGuard(Guardrails):
    name = "llamafirewall"

    def __init__(
        self,
        *,
        input_scanners: Sequence[str] = ("PROMPT_GUARD",),
        output_scanners: Sequence[str] = ("PROMPT_GUARD",),
        max_input_chars: int = 2000,
        block_on_human_review: bool = True,
        fail_open: bool = False,
        **_: Any,
    ) -> None:
        self.input_scanners = tuple(input_scanners)
        self.output_scanners = tuple(output_scanners)
        self.max_input_chars = max_input_chars
        self.block_on_human_review = block_on_human_review
        self.fail_open = fail_open
        self._module: Any = None
        self._firewall: Any = None

    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = require("llamafirewall", self.name)
        return self._module

    @property
    def firewall(self) -> Any:
        """Construct LlamaFirewall on first use; model downloads happen here."""
        if self._firewall is None:
            lf = self.module
            role, scanner_type = lf.Role, lf.ScannerType
            scanners: dict[Any, list[Any]] = {}
            if self.input_scanners:
                scanners[role.USER] = self._resolve(scanner_type, self.input_scanners)
            if self.output_scanners:
                scanners[role.ASSISTANT] = self._resolve(scanner_type, self.output_scanners)
            try:
                self._firewall = lf.LlamaFirewall(scanners=scanners)
            except ImportError as exc:
                raise AdapterUnavailable(self.name, "llamafirewall", str(exc)) from exc
        return self._firewall

    @staticmethod
    def _resolve(scanner_type: Any, names: Sequence[str]) -> list[Any]:
        """Map config strings like "PROMPT_GUARD" onto ScannerType members."""
        resolved = []
        for raw in names:
            member = getattr(scanner_type, str(raw).upper().replace("-", "_"), None)
            if member is None:
                known = [n for n in dir(scanner_type) if n.isupper()]
                raise ValueError(
                    f"Unknown LlamaFirewall scanner {raw!r}. Known: {', '.join(sorted(known))}"
                )
            resolved.append(member)
        return resolved

    def check_input(self, text: str) -> GuardResult:
        stripped = (text or "").strip()

        if not stripped:
            return GuardResult(
                allowed=False, reason="Input is empty.", validator="llamafirewall:min-length"
            )

        if len(stripped) > self.max_input_chars:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Input is {len(stripped)} characters, over the "
                    f"{self.max_input_chars}-character limit."
                ),
                validator="llamafirewall:max-length",
            )

        try:
            message = self.module.UserMessage(content=stripped)
            result = self.firewall.scan(message)
        except AdapterUnavailable:
            raise
        except Exception as exc:
            return self._unavailable(stripped, exc, "llamafirewall:input")

        return self._verdict(result, stripped, "llamafirewall:input")

    def check_output(self, text: str, context: str = "") -> GuardResult:
        out = text or ""

        if not self.output_scanners:
            return GuardResult(
                allowed=True,
                text=out,
                reason="No output scanners configured.",
                validator="llamafirewall:output",
            )

        try:
            message = self.module.AssistantMessage(content=out)
            result = self.firewall.scan(message)
        except AdapterUnavailable:
            raise
        except Exception as exc:
            return self._unavailable(out, exc, "llamafirewall:output")

        return self._verdict(result, out, "llamafirewall:output")

    def _verdict(self, result: Any, text: str, validator: str) -> GuardResult:
        """Translate a ScanResult into a GuardResult."""
        decision = getattr(result, "decision", None)
        label = getattr(decision, "name", str(decision))
        score = getattr(result, "score", None)
        reason = str(getattr(result, "reason", "") or "")

        blocked = label == "BLOCK"
        if self.block_on_human_review and label == "HUMAN_IN_THE_LOOP_REQUIRED":
            blocked = True

        if blocked:
            detail = f"LlamaFirewall decision {label}"
            if score is not None:
                detail += f" (score {score})"
            if reason:
                detail += f": {reason}"
            return GuardResult(allowed=False, reason=detail, validator=validator)

        return GuardResult(allowed=True, text=text, reason=reason, validator=validator)

    def _unavailable(self, text: str, exc: Exception, validator: str) -> GuardResult:
        """Scanners could not run. Fail closed unless explicitly told otherwise."""
        reason = (
            f"LlamaFirewall could not scan ({type(exc).__name__}: {exc}). "
            "Model weights may be missing -- run `llamafirewall configure`."
        )
        if self.fail_open:
            return GuardResult(
                allowed=True, text=text, reason=f"{reason} fail_open=True.", validator=validator
            )
        return GuardResult(allowed=False, reason=reason, validator=validator)
