"""L3 alternative: CrewAI (MIT).

CrewAI is the only framework in this level that does not model a turn as a graph
at all. It models it as a *role*: you describe who the worker is (an `Agent` with
a role, a goal and a backstory), what you want back (a `Task` with an
`expected_output`), and CrewAI runs the reasoning loop -- decide, call a tool,
read the result, decide again -- until the task is satisfied.

Against LangGraph that is a genuine trade, not just a different syntax:

  * LangGraph's edges are fixed at build time. You know exactly which steps will
    run and in what order, which is why its traces are so readable.
  * CrewAI's control flow is decided by the model at run time. The agent chooses
    whether to search once, search again with better terms, or answer directly.
    You gain adaptivity and lose determinism -- the same question can take a
    different number of steps on two consecutive runs.

That non-determinism is why this adapter pre-warms retrieval before the crew
starts (see `_search`): the citation numbering a run produces has to be stable,
and every other L3 adapter does exactly one retrieval per turn, so the swap demo
stays an apples-to-apples comparison.

The L0 and L4 levels stay swappable as always: `ArcLLM` subclasses CrewAI's
`BaseLLM` over `deps.inference` and the `knowledge_search` tool reads
`deps.store` + `deps.embeddings`. No CrewAI knowledge source or RAG tool is used.
Subclassing `BaseLLM` rather than `crewai.LLM` also keeps LiteLLM out of the call
path -- CrewAI 1.x does not even install it, so this adapter reaches a real agent
loop with no vendor SDK anywhere in the process.

Two practical notes. The first import of `crewai` takes roughly twenty seconds
and pulls in some 2,300 modules, so the pause before the first answer is not a
hang. And CrewAI wants to show an interactive traces panel after the first
kickoff, which `run` disables below -- on a real terminal it would otherwise
block on `input()` and take the whole CLI with it.
"""

from __future__ import annotations

import os
from typing import Any

from ... import rag_core
from ...citations import format_context
from ...interfaces import AgentDeps, AgentFramework
from ...registry import require
from ...types import Answer

_PKG = "crewai"
_ADAPTER = "crewai"

_BACKSTORY = (
    "You are a meticulous research assistant. You never state a fact that is not "
    "present in a retrieved passage, and you end every factual sentence with the "
    "[n] marker of the passage it came from. When the passages do not answer the "
    "question you say so plainly instead of guessing."
)

_TASK_DESCRIPTION = (
    "{memories}"
    "Answer this question: {question}\n\n"
    "First call the knowledge_search tool with the question to fetch the numbered "
    "context passages. Then answer using only those passages, ending every factual "
    "sentence with the matching [n] marker. If they do not contain the answer, "
    "reply exactly: I don't know based on the provided documents."
)

_EXPECTED_OUTPUT = (
    "A concise answer grounded in the retrieved passages, with a [n] citation "
    "marker at the end of every factual sentence."
)


class CrewAIAgent(AgentFramework):
    """Runs the turn as a one-agent crew with retrieval exposed as a CrewAI tool."""

    name = "crewai"

    def __init__(self, max_iter: int = 5, verbose: bool = False, **_: object) -> None:
        self.max_iter = max_iter
        self.verbose = verbose

    def run(self, question: str, deps: AgentDeps) -> Answer:
        # Telemetry has to be disabled before the package is imported, or CrewAI has
        # already installed its exporter. Arc Rector runs entirely on your machine.
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        # Without this, CrewAI 1.x opens an interactive input() prompt for its
        # "Execution Traces" panel after the first kickoff, which hangs the CLI on a
        # real terminal. Neither the telemetry flag, Crew(tracing=False), nor
        # CREWAI_TRACING_ENABLED suppresses it -- only this one does.
        os.environ.setdefault("CREWAI_TESTING", "true")
        # CrewAI still reaches for an OpenAI key in a few code paths even when every
        # model is local. setdefault never overwrites a real one you have exported.
        os.environ.setdefault("OPENAI_API_KEY", "arc-rector-unused")

        crewai = require(_PKG, _ADAPTER, "crewai")
        crew_tools = require(_PKG, _ADAPTER, "crewai.tools")
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

            # Per-turn retrieval state. The agent may call the tool more than once;
            # the first call decides the numbering and later calls reuse it, so a
            # marker never means two different passages within one answer.
            state: dict[str, Any] = {"hits": [], "citations": [], "context": ""}

            def _search(query: str) -> str:
                if state["context"]:
                    return state["context"]
                with tracer.span("retrieve", top_k=deps.top_k, fetch_k=deps.fetch_k) as sp:
                    hits = rag_core.retrieve(deps, query)
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
                state["hits"] = hits
                state["citations"] = citations
                state["context"] = context
                return context or "(no documents retrieved)"

            @crew_tools.tool("knowledge_search")
            def knowledge_search(query: str) -> str:
                """Search the indexed corpus and return numbered context passages.

                Each passage is prefixed with its citation marker, like [1] or [2].
                Cite those markers in your answer.
                """
                return _search(query)

            class ArcLLM(crewai.BaseLLM):
                """The L0 inference adapter, wearing CrewAI's LLM interface.

                Subclassing `BaseLLM` rather than `crewai.LLM` is what bypasses
                LiteLLM: CrewAI never tries to authenticate against a provider.
                """

                # `deps` comes from the enclosing scope: BaseLLM is a pydantic model
                # and silently swallows undeclared constructor kwargs into
                # `additional_params`, so passing deps in would quietly vanish.

                def call(
                    self,
                    messages: Any,
                    tools: Any = None,
                    callbacks: Any = None,
                    available_functions: Any = None,
                    **kwargs: Any,
                ) -> str:
                    # **kwargs absorbs from_task / from_agent / response_model, which
                    # CrewAI passes positionally-by-name and which grow between releases.
                    if isinstance(messages, str):
                        messages = [{"role": "user", "content": messages}]
                    system = "\n\n".join(
                        str(m.get("content", ""))
                        for m in messages
                        if m.get("role") == "system"
                    )
                    prompt = "\n\n".join(
                        str(m.get("content", ""))
                        for m in messages
                        if m.get("role") != "system"
                    )
                    with tracer.span(
                        "generate",
                        as_type="generation",
                        model=getattr(deps.inference, "model", ""),
                        input=prompt,
                    ) as sp:
                        text = deps.inference.complete(prompt, system=system or rag_core.SYSTEM_PROMPT)
                        sp.update(output=text)
                    return text

                def supports_function_calling(self) -> bool:
                    # A plain complete() backend has no tool-call protocol, so CrewAI
                    # must fall back to its text-based ReAct loop.
                    return False

                def supports_stop_words(self) -> bool:
                    return False

                def get_context_window_size(self) -> int:
                    return 8192

            llm = ArcLLM(model=str(getattr(deps.inference, "model", "arc-rector-l0")), temperature=0.0)

            agent = crewai.Agent(
                role="Grounded research assistant",
                goal="Answer the user's question using only the indexed corpus, with citations",
                backstory=_BACKSTORY,
                llm=llm,
                tools=[knowledge_search],
                allow_delegation=False,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )
            task = crewai.Task(
                description=_TASK_DESCRIPTION,
                expected_output=_EXPECTED_OUTPUT,
                agent=agent,
            )
            crew = crewai.Crew(
                agents=[agent],
                tasks=[task],
                process=crewai.Process.sequential,
                verbose=self.verbose,
            )

            # Retrieve once up front so the numbering is fixed before the agent can
            # reason about it; its own tool call then hits the cached result.
            _search(question)

            with tracer.span("crew.kickoff", agents=1, tasks=1) as sp:
                # Values go through `inputs` rather than f-strings so that braces in
                # the question or in a remembered fact are never treated as slots.
                result = crew.kickoff(
                    inputs={
                        "question": question,
                        "memories": rag_core.format_memories(memories),
                    }
                )
                raw = str(getattr(result, "raw", "") or result)
                sp.update(output=raw)

            hits = state["hits"]
            citations = state["citations"]
            context = state["context"]

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
