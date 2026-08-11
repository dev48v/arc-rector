"""L3 default: LangGraph (MIT).

LangGraph models the turn as an explicit state machine. That buys three things
this project cares about:

  * every step is a named node, so a trace reads like the graph rather than like
    a stack trace;
  * conditional edges express "guardrail rejected -> stop" without an early
    return buried in a function;
  * the graph is inspectable and checkpointable, which is what you want once a
    turn grows a retry or a human-in-the-loop pause.

The nodes below call into `arc_rector.rag_core`, so the retrieval and prompting
are identical to every other L3 adapter. Only the orchestration differs.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ... import rag_core
from ...citations import format_context
from ...interfaces import AgentDeps, AgentFramework
from ...registry import require
from ...types import Answer, Citation, MemoryRecord, Retrieved


class TurnState(TypedDict, total=False):
    """State threaded through the graph. LangGraph merges node returns into it."""

    question: str
    memories: list[MemoryRecord]
    hits: list[Retrieved]
    context: str
    citations: list[Citation]
    warning: str
    raw: str
    answer: Answer
    blocked: bool
    block_reason: str
    block_validator: str


class LangGraphAgent(AgentFramework):
    """Compiles a StateGraph: guard_in -> recall -> retrieve -> generate -> guard_out -> remember."""

    name = "langgraph"

    def __init__(self, **_: object) -> None:
        self._graph: Any = None

    def _build_graph(self, deps: AgentDeps) -> Any:
        graph_mod = require("langgraph", "langgraph", "langgraph.graph")
        StateGraph = graph_mod.StateGraph
        END = graph_mod.END
        tracer = deps.tracer

        def guard_input(state: TurnState) -> dict[str, Any]:
            with tracer.span("guardrails.input", text=state["question"]) as sp:
                res = deps.guardrails.check_input(state["question"])
                sp.update(output={"allowed": res.allowed, "reason": res.reason})
            if not res.allowed:
                return {"blocked": True, "block_reason": res.reason, "block_validator": res.validator}
            return {"blocked": False}

        def recall(state: TurnState) -> dict[str, Any]:
            with tracer.span("memory.recall", user_id=deps.user_id) as sp:
                memories = rag_core.recall(deps, state["question"])
                sp.update(output=[m.text for m in memories])
            return {"memories": memories}

        def retrieve(state: TurnState) -> dict[str, Any]:
            with tracer.span("retrieve", top_k=deps.top_k, fetch_k=deps.fetch_k) as sp:
                hits = rag_core.retrieve(deps, state["question"])
                sp.update(
                    output=[
                        {"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "title": h.chunk.title}
                        for h in hits
                    ]
                )
            context, citations = format_context(hits, deps.max_context_chars)
            context, warning = rag_core.guard_context(deps, context)
            return {"hits": hits, "context": context, "citations": citations, "warning": warning}

        def generate(state: TurnState) -> dict[str, Any]:
            prompt = rag_core.build_prompt(
                state["question"],
                state.get("context", ""),
                state.get("memories", []),
                state.get("warning", ""),
            )
            with tracer.span(
                "generate", as_type="generation", model=getattr(deps.inference, "model", ""), input=prompt
            ) as sp:
                raw = deps.inference.complete(prompt, system=rag_core.SYSTEM_PROMPT)
                sp.update(output=raw)
            return {"raw": raw}

        def guard_output(state: TurnState) -> dict[str, Any]:
            with tracer.span("guardrails.output") as sp:
                res = deps.guardrails.check_output(state.get("raw", ""), context=state.get("context", ""))
                sp.update(output={"allowed": res.allowed, "reason": res.reason})
            if not res.allowed:
                return {"blocked": True, "block_reason": res.reason, "block_validator": res.validator}
            return {"raw": res.text or state.get("raw", "")}

        def remember(state: TurnState) -> dict[str, Any]:
            answer = rag_core.assemble(
                state["question"],
                state.get("raw", ""),
                state.get("hits", []),
                state.get("citations", []),
                state.get("memories", []),
            )
            with tracer.span("memory.write", user_id=deps.user_id):
                rag_core.remember(deps, state["question"], answer.text)
            return {"answer": answer}

        def finish_blocked(state: TurnState) -> dict[str, Any]:
            return {
                "answer": rag_core.blocked_answer(
                    state["question"], state.get("block_reason", ""), state.get("block_validator", "")
                )
            }

        # Conditional edges are the reason to use a graph rather than an if-chain:
        # the rejection path is a first-class edge, visible in the compiled graph.
        def route(state: TurnState) -> str:
            return "blocked" if state.get("blocked") else "ok"

        builder = StateGraph(TurnState)
        builder.add_node("guard_input", guard_input)
        builder.add_node("recall", recall)
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)
        builder.add_node("guard_output", guard_output)
        builder.add_node("remember", remember)
        builder.add_node("blocked", finish_blocked)

        builder.set_entry_point("guard_input")
        builder.add_conditional_edges("guard_input", route, {"ok": "recall", "blocked": "blocked"})
        builder.add_edge("recall", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "guard_output")
        builder.add_conditional_edges("guard_output", route, {"ok": "remember", "blocked": "blocked"})
        builder.add_edge("remember", END)
        builder.add_edge("blocked", END)
        return builder.compile()

    def run(self, question: str, deps: AgentDeps) -> Answer:
        # Rebuilt per run because the closures capture `deps`, which can change
        # when the caller swaps a level between turns.
        self._graph = self._build_graph(deps)
        with deps.tracer.span("arc-rector.turn", framework=self.name, question=question) as turn:
            final: TurnState = self._graph.invoke({"question": question})
            answer: Answer = final.get("answer") or rag_core.blocked_answer(
                question, "graph produced no answer"
            )
            turn.update(
                output={
                    "answer": answer.text,
                    "blocked": answer.blocked,
                    "citations": len(answer.citations),
                }
            )
            answer.trace_id = deps.tracer.last_trace_id()
            return answer

    def draw(self, deps: AgentDeps) -> str:
        """Mermaid source for the compiled graph -- used to keep the README honest."""
        graph = self._graph or self._build_graph(deps)
        return graph.get_graph().draw_mermaid()
