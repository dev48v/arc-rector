"""L3 alternative: LlamaIndex (MIT).

LlamaIndex is a retrieval framework first and an orchestration framework second.
Where LangGraph hands you a state machine and leaves retrieval entirely to you,
LlamaIndex hands you the *query pipeline* -- a retriever, optional node
postprocessors, and a response synthesizer that knows how to pack, compact and
refine chunks into one or more model calls -- and leaves control flow to you.

So the two frameworks are strong in opposite places, and this adapter shows it:
there are no conditional edges here, but there is also no hand-written "stuff
the chunks into a string" step, because `RetrieverQueryEngine` owns that.

The important part for Arc Rector is what is *absent*. LlamaIndex ships its own
vector store integrations and its own LLM integrations, and this adapter uses
none of them:

  * `ArcStoreRetriever` subclasses `BaseRetriever` and answers `_retrieve` out of
    `deps.store` + `deps.embeddings`, so the L4 and L5 rows of config.yaml stay
    in charge of the vector DB and the embedding model;
  * `ArcInferenceLLM` subclasses `CustomLLM` and answers `complete` out of
    `deps.inference`, so the L0 row stays in charge of the model.

That is the whole thesis of the project in one file: the framework is swappable
independently of the vector database, because the framework only ever sees an
interface we implement.
"""

from __future__ import annotations

from typing import Any

from ... import rag_core
from ...citations import format_context
from ...interfaces import AgentDeps, AgentFramework
from ...registry import require
from ...types import Answer, Chunk, Citation

_PKG = "llama-index-core"
_ADAPTER = "llamaindex"

# `{memory_block}` is supplied via partial_format rather than string concatenation
# so that a remembered fact containing braces can never be reparsed as a
# template variable. `{context_str}` and `{query_str}` are LlamaIndex's own names.
_QA_TEMPLATE = (
    "{memory_block}"
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer using only the context above, with [n] citation markers:"
)


def _numbered(cite: Citation, chunk: Chunk) -> str:
    """Render one context entry with the same header shape `format_context` uses.

    The marker has to be inside the node's own text: LlamaIndex builds
    `{context_str}` from node content, so a marker kept only in node metadata
    would never reach the model and every citation would dangle.
    """
    header = f"[{cite.marker}] {cite.title}"
    if cite.source:
        header += f" -- {cite.source}"
    return f"{header}\n{chunk.text.strip()}"


class LlamaIndexAgent(AgentFramework):
    """Runs the turn through a LlamaIndex RetrieverQueryEngine over Arc Rector's own store."""

    name = "llamaindex"

    def __init__(self, response_mode: str = "compact", context_window: int = 8192, **_: object) -> None:
        self.response_mode = response_mode
        self.context_window = context_window

    def _components(self, deps: AgentDeps) -> tuple[Any, Any]:
        """Build the retriever and the LLM.

        Both classes are defined in here rather than at module scope so that
        importing this module never requires llama-index to be installed --
        `arc-rector levels` has to be able to list this adapter on a machine that
        has never heard of LlamaIndex.
        """
        llms = require(_PKG, _ADAPTER, "llama_index.core.llms")
        callbacks = require(_PKG, _ADAPTER, "llama_index.core.llms.callbacks")
        schema = require(_PKG, _ADAPTER, "llama_index.core.schema")
        retrievers = require(_PKG, _ADAPTER, "llama_index.core.retrievers")

        context_window = self.context_window

        class ArcInferenceLLM(llms.CustomLLM):
            """The L0 inference adapter, wearing LlamaIndex's LLM interface."""

            # `deps` is captured from the enclosing scope on purpose: CustomLLM is a
            # pydantic model, so assigning an *undeclared* instance attribute in
            # __init__ raises "object has no field". Declared fields and PrivateAttrs
            # both work fine -- but a PrivateAttr set before super().__init__() is
            # silently reset to None, which is a much worse bug to chase than this
            # closure. Contrast ArcStoreRetriever below, which is not a pydantic model.

            @property
            def metadata(self) -> Any:
                return llms.LLMMetadata(
                    context_window=context_window,
                    num_output=1024,
                    model_name=str(getattr(deps.inference, "model", "arc-rector-l0")),
                    is_chat_model=False,
                )

            @callbacks.llm_completion_callback()
            def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
                with deps.tracer.span(
                    "generate",
                    as_type="generation",
                    model=getattr(deps.inference, "model", ""),
                    input=prompt,
                ) as sp:
                    text = deps.inference.complete(prompt, system=rag_core.SYSTEM_PROMPT)
                    sp.update(output=text)
                return llms.CompletionResponse(text=text)

            @callbacks.llm_completion_callback()
            def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
                # Abstract on CustomLLM, so it must exist even though the CLI never streams.
                text = deps.inference.complete(prompt, system=rag_core.SYSTEM_PROMPT)
                yield llms.CompletionResponse(text=text, delta=text)

        class ArcStoreRetriever(retrievers.BaseRetriever):
            """The L4 store + L5 embeddings, wearing LlamaIndex's retriever interface.

            This is the swap point. No llama-index vector store integration is
            installed or imported, so switching L4 from qdrant to chroma in
            config.yaml changes nothing in this file.
            """

            def __init__(self) -> None:
                self.hits: list[Any] = []
                self.citations: list[Citation] = []
                self.context: str = ""
                # BaseRetriever is a plain class, not a pydantic model, so assigning
                # before super() is safe here -- the exact opposite of the LLM above.
                # super() still has to run: it installs callback_manager and object_map.
                super().__init__()

            def _retrieve(self, query_bundle: Any) -> list[Any]:
                with deps.tracer.span("retrieve", top_k=deps.top_k, fetch_k=deps.fetch_k) as sp:
                    hits = rag_core.retrieve(deps, query_bundle.query_str)
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

                # Citations are numbered once, here, from the shared helper -- the
                # nodes below carry those exact markers so the answer's [n] tie back.
                context, citations = format_context(hits, deps.max_context_chars)
                self.hits = hits
                self.citations = citations
                self.context = context

                by_id = {c.chunk_id: c for c in citations}
                nodes: list[Any] = []
                seen: set[str] = set()
                for hit in hits:
                    cite = by_id.get(hit.chunk.chunk_id)
                    if cite is None or cite.chunk_id in seen:
                        continue  # dropped by the context budget, or a repeat
                    seen.add(cite.chunk_id)
                    meta = {
                        "chunk_id": cite.chunk_id,
                        "marker": cite.marker,
                        "title": cite.title,
                        "source": cite.source,
                    }
                    node = schema.TextNode(
                        text=_numbered(cite, hit.chunk),
                        id_=cite.chunk_id,
                        metadata=meta,
                        # The marker lives in the node text, not in metadata, so the
                        # citations survive even if these exclusions are ever dropped.
                        # Excluding them just stops LlamaIndex prepending the same
                        # values again as "key: value" lines above every passage.
                        excluded_llm_metadata_keys=list(meta),
                        excluded_embed_metadata_keys=list(meta),
                    )
                    nodes.append(schema.NodeWithScore(node=node, score=hit.score))
                return nodes

        return ArcStoreRetriever(), ArcInferenceLLM()

    def run(self, question: str, deps: AgentDeps) -> Answer:
        prompts = require(_PKG, _ADAPTER, "llama_index.core.prompts")
        query_engine = require(_PKG, _ADAPTER, "llama_index.core.query_engine")
        synthesizers = require(_PKG, _ADAPTER, "llama_index.core.response_synthesizers")
        prompt_helper = require(_PKG, _ADAPTER, "llama_index.core.indices.prompt_helper")
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

            retriever, llm = self._components(deps)
            template = prompts.PromptTemplate(_QA_TEMPLATE).partial_format(
                memory_block=rag_core.format_memories(memories)
            )

            # The PromptHelper is built explicitly on purpose. `from_args` resolves it
            # as `Settings.prompt_helper or PromptHelper.from_llm_metadata(...)`, and
            # the Settings getter lazily returns a default 3900-token helper, so the
            # fallback is dead code: the context_window declared on ArcInferenceLLM
            # would be silently ignored and long contexts split into extra refine calls.
            synthesizer = synthesizers.get_response_synthesizer(
                llm=llm,
                response_mode=self.response_mode,
                text_qa_template=template,
                prompt_helper=prompt_helper.PromptHelper.from_llm_metadata(llm.metadata),
            )
            # Passing `llm` explicitly is what keeps Settings.llm untouched, so no
            # OpenAI key is ever needed on this path.
            engine = query_engine.RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=synthesizer,
            )

            with tracer.span("query_engine", response_mode=self.response_mode) as sp:
                response = engine.query(question)
                raw = str(response)
                sp.update(output={"answer": raw, "source_nodes": len(getattr(response, "source_nodes", []))})

            # Retrieval happened inside the engine, so the results come back off
            # the retriever we handed it.
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
