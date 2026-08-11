"""L0 offline option: a scripted, deterministic fake model.

There is no model here. It extracts the first sentences from the numbered context
block it was handed and echoes them back with the matching citation markers.

That is enough to exercise every other layer -- retrieval, citation formatting,
guardrails, memory, tracing -- with byte-identical output on every run, which is
what the test suite and `make demo-offline` need. It is also a useful control:
if the answer looks the same with `echo` as with a real model, your retrieval is
doing the work and the model is not adding anything.
"""

from __future__ import annotations

import re
from typing import Any

from ...interfaces import Inference

_ENTRY = re.compile(r"^\[(\d+)\]\s*(.*)$", re.MULTILINE)
_NO_ANSWER = "I don't know based on the provided documents."


class EchoInference(Inference):
    name = "echo"

    def __init__(self, *, max_sentences: int = 2, **_: Any) -> None:
        self.model = "echo"
        self.max_sentences = max_sentences

    def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        context = self._extract_context(prompt)
        entries = self._parse_entries(context)
        if not entries:
            return _NO_ANSWER

        parts: list[str] = []
        for marker, body in entries[: self.max_sentences]:
            sentence = self._first_sentence(body)
            if sentence:
                parts.append(f"{sentence} [{marker}]")
        return " ".join(parts) if parts else _NO_ANSWER

    @staticmethod
    def _extract_context(prompt: str) -> str:
        """Pull the block between 'Context:' and 'Question:' out of the prompt."""
        start = prompt.find("Context:")
        end = prompt.find("Question:")
        if start == -1:
            return ""
        body = prompt[start + len("Context:") : end if end > start else len(prompt)]
        return body.strip()

    @staticmethod
    def _parse_entries(context: str) -> list[tuple[int, str]]:
        """Split the numbered context into (marker, text) pairs."""
        matches = list(_ENTRY.finditer(context))
        out: list[tuple[int, str]] = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            body = context[start:end].strip()
            if body:
                out.append((int(match.group(1)), body))
        return out

    @staticmethod
    def _first_sentence(text: str) -> str:
        flat = " ".join(text.split())
        match = re.search(r"^(.{20,400}?[.!?])(\s|$)", flat)
        sentence = match.group(1) if match else flat[:200]
        return sentence.rstrip(".").strip() + "."

    def available(self) -> bool:
        return True
