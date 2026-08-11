"""L3 alternative: DSPy (MIT).

DSPy is the odd one out, and deliberately so. Every other framework at this level
takes the prompt you wrote and moves it around a graph. DSPy takes the prompt
*away from you*: you declare a `Signature` -- these are the inputs, this is the
output, here is what the task means -- and DSPy compiles that declaration into
whatever text the model actually sees, then parses the reply back into typed
fields.

Against LangGraph the difference is not orchestration, it is authorship:

  * LangGraph is prompt-agnostic. `rag_core.SYSTEM_PROMPT` is passed through
    untouched, and if the answer is badly formatted you edit that string.
  * DSPy owns the wire format. It appends its own field markers to the prompt and
    refuses replies it cannot parse. In exchange, the same `Signature` can later
    be *optimised* -- few-shot examples selected automatically against a metric --
    without rewriting anything here.

That is the honest trade to weigh: DSPy asks a lot more of the model's ability to
follow a structured output format, which a small local L0 model may not do
reliably, and pays you back with programs you can tune instead of prompts you
have to fiddle with.

Retrieval is a plain Python callable inside `forward()`. That is not a shortcut:
DSPy 3 removed its bundled retriever integrations, and a callable is now the
documented way to bring your own -- which suits Arc Rector exactly, because the
callable reads `deps.store` + `deps.embeddings` and leaves L4 and L5 swappable.
"""

from __future__ import annotations

import types as pytypes
from typing import Any

from ... import rag_core
from ...citations import format_context
from ...interfaces import AgentDeps, AgentFramework
from ...registry import require
from ...types import Answer

_PKG = "dspy"
_ADAPTER = "dspy"


class DspyAgent(AgentFramework):
    """Runs the turn as a DSPy Module: a Signature plus a bring-your-own retriever."""

    name = "dspy"

    def __init__(self, chain_of_thought: bool = False, **_: object) -> None:
        # dspy.ChainOfThought adds a `reasoning` field to the same Signature, so the
        # model must emit BOTH [[ ## reasoning ## ]] and [[ ## answer ## ]] markers --
        # returning only the answer still fails to parse. That is measurably more
        # brittle than Predict on the small local models this repo ships with, so it
        # is opt-in rather than the default.
        self.chain_of_thought = chain_of_thought

    def _program(self, deps: AgentDeps, dspy: Any) -> tuple[Any, Any]:
        """Build the LM wrapper and the RAG module. Returns (lm, module)."""

        class ArcLM(dspy.BaseLM):
            """The L0 inference adapter, wearing DSPy's LM interface.

            Subclassing `BaseLM` rather than `dspy.LM` is what bypasses LiteLLM:
            no provider string is parsed and no API key is ever looked up.
            """

            # "legacy" is the prompt/messages-in, OpenAI-shaped-response-out contract.
            # DSPy warns if a custom LM leaves it unset.
            forward_contract = "legacy"

            def __init__(self) -> None:
                super().__init__(
                    model=str(getattr(deps.inference, "model", "arc-rector-l0")),
                    model_type="chat",
                )

            def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
                system, user = _split_messages(prompt, messages)
                with deps.tracer.span(
                    "generate",
                    as_type="generation",
                    model=getattr(deps.inference, "model", ""),
                    input=user,
                ) as sp:
                    text = deps.inference.complete(user, system=system or rag_core.SYSTEM_PROMPT)
                    sp.update(output=text)
                # DSPy reads .choices[].message.content, .model and .usage off this.
                # A SimpleNamespace satisfies it; no LiteLLM types are needed.
                return pytypes.SimpleNamespace(
                    choices=[
                        pytypes.SimpleNamespace(
                            message=pytypes.SimpleNamespace(content=text, tool_calls=None)
                        )
                    ],
                    model=self.model,
                    usage={},
                )

        class GroundedAnswer(dspy.Signature):
            """Answer a question using only the numbered context passages provided.

            End every factual sentence with the citation marker of the passage it
            came from, like [1] or [2]. Never invent a marker that is not in the
            context. If the context does not contain the answer, reply exactly:
            I don't know based on the provided documents.
            """

            context: str = dspy.InputField(desc="numbered context passages, each starting with [n]")
            memories: str = dspy.InputField(desc="what is already known about this user; may be empty")
            question: str = dspy.InputField(desc="the user's question")
            answer: str = dspy.OutputField(desc="a concise answer with [n] citation markers")

        # A Signature's docstring is its instructions, and DSPy compiles those into
        # the prompt. Overriding them with the shared SYSTEM_PROMPT keeps the wording
        # of the task identical to every other L3 adapter, so the swap demo compares
        # orchestration rather than prompt luck.
        signature = GroundedAnswer.with_instructions(rag_core.SYSTEM_PROMPT)
        predictor = dspy.ChainOfThought if self.chain_of_thought else dspy.Predict

        class ArcRag(dspy.Module):
            """Retrieve, then predict. The two-line version of this whole project."""

            def __init__(self) -> None:
                super().__init__()
                self.respond = predictor(signature)
                self.hits: list[Any] = []
                self.citations: list[Any] = []
                self.context: str = ""

            def forward(self, question: str, memories: str = "") -> Any:
                with deps.tracer.span("retrieve", top_k=deps.top_k, fetch_k=deps.fetch_k) as sp:
                    hits = rag_core.retrieve(deps, question)
                    sp.update(
                        output=[
                            {
                                "chunk_id": h.chunk.chunk_id,
                                "score": round(h.score, 4),
                                "title": h.chunk.title,
                            }
                            for h in hits
                        ]
                    )
                context, citations = format_context(hits, deps.max_context_chars)
                self.hits = hits
                self.citations = citations
                self.context = context
                return self.respond(
                    context=context or "(no documents retrieved)",
                    memories=memories,
                    question=question,
                )

        return ArcLM(), ArcRag()

    def run(self, question: str, deps: AgentDeps) -> Answer:
        dspy = require(_PKG, _ADAPTER)
        dspy_exceptions = require(_PKG, _ADAPTER, "dspy.utils.exceptions")
        tracer = deps.tracer

        with tracer.span("arc-rector.turn", framework=self.name, question=question) as turn:

            with tracer.span("guardrails.input", text=question) as sp:
                guard_in = deps.guardrails.check_input(question)
                sp.update(output={"allowed": guard_in.allowed, "reason": guard_in.reason})
            if not guard_in.allowed:
                answer = rag_core.blocked_answer(question, guard_in.reason, guard_in.validator)
                turn.update(output={"blocked": True, "reason": answer.block_reason})
                return answer

            with tracer.span("memory.recall", user_id=deps.user_id) as sp:
                memories = rag_core.recall(deps, question)
                sp.update(output=[m.text for m in memories])

            lm, module = self._program(deps, dspy)

            with tracer.span("dspy.module", predictor=type(module.respond).__name__) as sp:
                try:
                    # dspy.context scopes the LM to this call instead of mutating global
                    # settings, so two adapters can run in one process without racing.
                    # Expect up to TWO L0 calls per turn: if ChatAdapter cannot parse
                    # the reply, DSPy silently retries the whole thing with JSONAdapter
                    # and a different prompt format before giving up. The second call in
                    # a trace is that retry, not a bug.
                    with dspy.context(lm=lm):
                        prediction = module(
                            question=question,
                            memories=rag_core.format_memories(memories),
                        )
                except dspy_exceptions.AdapterParseError as exc:
                    # The trade named at the top of this file, arriving in person: DSPy
                    # owns the wire format, so a model that will not emit its field
                    # markers produces no answer at all rather than a sloppy one.
                    sp.update(output={"parse_error": str(exc)[:500]})
                    answer = rag_core.blocked_answer(
                        question,
                        "DSPy could not parse the model's reply into the answer field. "
                        "Small local models often miss its [[ ## field ## ]] markers -- "
                        "try a larger l0_inference model, or l3_framework: simple",
                        validator="dspy.AdapterParseError",
                    )
                    answer.retrieved = list(module.hits)
                    answer.memories_used = [m.text for m in memories]
                    turn.update(output={"blocked": True, "reason": answer.block_reason})
                    answer.trace_id = tracer.last_trace_id()
                    return answer
                raw = str(getattr(prediction, "answer", "") or "")
                sp.update(output=raw)

            hits = module.hits
            citations = module.citations
            context = module.context

            with tracer.span("guardrails.output") as sp:
                guard_out = deps.guardrails.check_output(raw, context=context)
                sp.update(output={"allowed": guard_out.allowed, "reason": guard_out.reason})
            if not guard_out.allowed:
                answer = rag_core.blocked_answer(question, guard_out.reason, guard_out.validator)
                turn.update(output={"blocked": True, "reason": answer.block_reason})
                return answer

            answer = rag_core.assemble(question, guard_out.text or raw, hits, citations, memories)

            with tracer.span("memory.write", user_id=deps.user_id):
                rag_core.remember(deps, question, answer.text)

            turn.update(output={"answer": answer.text, "citations": len(answer.citations)})
            answer.trace_id = tracer.last_trace_id()
            return answer


def _split_messages(prompt: Any, messages: Any) -> tuple[str, str]:
    """Flatten DSPy's chat messages into the (system, user) pair L0 expects.

    DSPy puts its compiled Signature instructions in the system role and the
    populated input fields in the user role, so this is where DSPy's authorship of
    the prompt actually takes effect.
    """
    if not messages:
        return "", str(prompt or "")
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        if message.get("role") in ("system", "developer"):
            system_parts.append(content)
        else:
            user_parts.append(content)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)
