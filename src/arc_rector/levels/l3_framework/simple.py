"""L3 reference implementation: no agent framework at all.

This is the control group. It runs exactly the same retrieve -> reason -> answer
turn as the LangGraph adapter, with zero orchestration dependencies, so you can
prove what a framework is actually buying you. If `simple` and `langgraph`
produce comparably good answers on your corpus, the framework is earning its
keep on ergonomics and observability rather than on answer quality -- which is
the honest thing to know before adopting one.

Every other L3 adapter mirrors the step sequence below.
"""

from __future__ import annotations

from ... import rag_core
from ...interfaces import AgentDeps, AgentFramework
from ...citations import format_context
from ...types import Answer


class SimpleAgent(AgentFramework):
    """Straight-line RAG: guard -> recall -> retrieve -> generate -> guard -> remember."""

    name = "simple"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, question: str, deps: AgentDeps) -> Answer:
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

            with tracer.span("retrieve", top_k=deps.top_k, fetch_k=deps.fetch_k) as sp:
                hits = rag_core.retrieve(deps, question)
                sp.update(
                    output=[
                        {"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "title": h.chunk.title}
                        for h in hits
                    ]
                )

            context, citations = format_context(hits, deps.max_context_chars)
            prompt = rag_core.build_prompt(question, context, memories)

            with tracer.span(
                "generate", as_type="generation", model=getattr(deps.inference, "model", ""), input=prompt
            ) as sp:
                raw = deps.inference.complete(prompt, system=rag_core.SYSTEM_PROMPT)
                sp.update(output=raw)

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
