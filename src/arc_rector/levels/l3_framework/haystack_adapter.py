"""L3 alternative: Haystack (Apache-2.0).

Haystack models a turn as a *dataflow graph* rather than a state machine. That
is the real difference from LangGraph, and it is worth being precise about it:

  * LangGraph nodes all read and write one shared state dict, and edges say
    "what runs next". Branching is natural; a node can see anything any earlier
    node wrote.
  * Haystack components declare typed input and output *sockets*, and edges say
    "this output feeds that input". Nothing is shared implicitly. A pipeline that
    type-checks at `connect()` time is a pipeline whose wiring is correct before
    it ever runs -- you find out you connected `documents` to `prompt` when you
    build it, not three minutes into a live call.

The trade is symmetric: Haystack catches wiring mistakes early and makes the
prompt assembly a first-class, swappable component (`PromptBuilder` with a Jinja
template), while LangGraph makes conditional control flow easier to express.

As everywhere else in Arc Rector, the framework's own integrations are unused:
`ArcRetriever` is a hand-written `@component` over `deps.store` and
`deps.embeddings` rather than a Haystack document store, and `ArcGenerator` is a
`@component` over `deps.inference` rather than a Haystack generator. Swapping L4
or L0 in config.yaml therefore never touches this file.
"""

from __future__ import annotations

from typing import Any

from ... import rag_core
from ...citations import format_context
from ...interfaces import AgentDeps, AgentFramework
from ...registry import require
from ...types import Answer, Citation

_PKG = "haystack-ai"
_ADAPTER = "haystack"

# Prompt assembly as a Jinja template is the Haystack idiom: the wording of the
# turn lives in data, not in Python, so it can be edited without touching code.
# It deliberately mirrors `rag_core.build_prompt` so answers stay comparable
# across every L3 adapter.
_PROMPT_TEMPLATE = """{% if memories %}What you already know about this user:
{% for fact in memories %}- {{ fact }}
{% endfor %}
{% endif %}Context:
{% for doc in documents %}[{{ doc.meta["marker"] }}] {{ doc.meta["title"] }}{% if doc.meta["source"] %} -- {{ doc.meta["source"] }}{% endif %}
{{ doc.content }}
{% endfor %}
Question: {{ question }}

Answer using only the context above, with [n] citation markers:"""


class HaystackAgent(AgentFramework):
    """Runs the turn through a Haystack Pipeline: retrieve -> build prompt -> generate."""

    name = "haystack"

    def __init__(self, **_: object) -> None:
        pass

    def _build_pipeline(self, deps: AgentDeps, haystack: Any, builders: Any) -> tuple[Any, Any]:
        """Wire the three components. Returns the pipeline and the retriever.

        The component classes are defined in here rather than at module scope so
        that importing this module never requires haystack to be installed: the
        `@component` decorator itself has to exist before a class can use it.
        """
        component = haystack.component
        Document = haystack.Document

        @component
        class ArcRetriever:
            """The L4 store + L5 embeddings, wearing Haystack's component interface."""

            def __init__(self) -> None:
                self.hits: list[Any] = []
                self.citations: list[Citation] = []
                self.context: str = ""

            @component.output_types(documents=list[Document])
            def run(self, question: str) -> dict[str, Any]:
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

                # Numbering happens once, in the shared helper, and rides along in
                # `meta` so the Jinja template and the returned Citations agree.
                context, citations = format_context(hits, deps.max_context_chars)
                self.hits = hits
                self.citations = citations
                self.context = context

                by_id = {c.chunk_id: c for c in citations}
                documents: list[Any] = []
                seen: set[str] = set()
                for hit in hits:
                    cite = by_id.get(hit.chunk.chunk_id)
                    if cite is None or cite.chunk_id in seen:
                        continue  # dropped by the context budget, or a repeat
                    seen.add(cite.chunk_id)
                    documents.append(
                        Document(
                            content=hit.chunk.text.strip(),
                            meta={
                                "marker": cite.marker,
                                "title": cite.title,
                                "source": cite.source,
                                "chunk_id": cite.chunk_id,
                            },
                            score=hit.score,
                        )
                    )
                return {"documents": documents}

        @component
        class ArcGenerator:
            """The L0 inference adapter, wearing Haystack's component interface."""

            @component.output_types(replies=list[str])
            def run(self, prompt: str) -> dict[str, Any]:
                with deps.tracer.span(
                    "generate",
                    as_type="generation",
                    model=getattr(deps.inference, "model", ""),
                    input=prompt,
                ) as sp:
                    text = deps.inference.complete(prompt, system=rag_core.SYSTEM_PROMPT)
                    sp.update(output=text)
                return {"replies": [text]}

        retriever = ArcRetriever()
        pipeline = haystack.Pipeline()
        pipeline.add_component("retrieve", retriever)
        pipeline.add_component("prompt", builders.PromptBuilder(template=_PROMPT_TEMPLATE))
        pipeline.add_component("generate", ArcGenerator())

        # Socket-to-socket wiring, validated here rather than at run time.
        pipeline.connect("retrieve.documents", "prompt.documents")
        pipeline.connect("prompt.prompt", "generate.prompt")
        return pipeline, retriever

    def run(self, question: str, deps: AgentDeps) -> Answer:
        haystack = require(_PKG, _ADAPTER, "haystack")
        builders = require(_PKG, _ADAPTER, "haystack.components.builders")
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

            pipeline, retriever = self._build_pipeline(deps, haystack, builders)

            with tracer.span("pipeline.run", components=["retrieve", "prompt", "generate"]) as sp:
                # `question` is fed to two components: sockets are not shared state,
                # so anything two components both need has to be handed to both.
                result = pipeline.run(
                    {
                        "retrieve": {"question": question},
                        "prompt": {
                            "question": question,
                            "memories": [m.text for m in memories],
                        },
                    }
                )
                sp.update(output={"components": sorted(result)})

            raw = result["generate"]["replies"][0]
            hits = retriever.hits
            citations = retriever.citations
            context = retriever.context

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
