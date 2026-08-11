"""The RAG steps every L3 framework adapter shares.

The framework layer is supposed to be swappable, so the *substance* of a turn --
what gets retrieved, what the prompt says, how citations are attached -- lives
here, and each framework adapter only supplies the orchestration idiom
(a LangGraph StateGraph, a Haystack Pipeline, a DSPy Module, and so on).

That split is what makes the swap demo meaningful: change L3 and the answer
quality stays the same, because only the plumbing changed.
"""

from __future__ import annotations

from typing import Sequence

from .citations import format_context, prune_citations
from .interfaces import AgentDeps
from .types import Answer, Citation, MemoryRecord, Retrieved

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer ONLY from the numbered context "
    "provided. Every factual sentence must end with a citation marker like [1] or "
    "[2] identifying the context entry it came from. If the context does not "
    "contain the answer, say exactly: I don't know based on the provided documents. "
    "Never invent a citation number that is not in the context. Be concise."
)


def retrieve(deps: AgentDeps, question: str) -> list[Retrieved]:
    """Embed the question, pull `fetch_k` candidates, rerank down to `top_k`."""
    vector = deps.embeddings.embed_query(question)
    hits = deps.store.search(vector, top_k=deps.fetch_k)
    return deps.reranker.rerank(question, hits, deps.top_k)


def recall(deps: AgentDeps, question: str) -> list[MemoryRecord]:
    """Pull long-term memories relevant to this question for this user."""
    try:
        return deps.memory.search(question, deps.user_id, top_k=3)
    except Exception:
        # Memory is an enhancement, never a hard dependency of answering.
        return []


def format_memories(memories: Sequence[MemoryRecord]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m.text}" for m in memories)
    return f"What you already know about this user:\n{lines}\n\n"


def build_prompt(question: str, context: str, memories: Sequence[MemoryRecord] = ()) -> str:
    """Assemble the final user-turn prompt sent to whichever L0 model is active."""
    memory_block = format_memories(memories)
    return (
        f"{memory_block}"
        f"Context:\n{context if context else '(no documents retrieved)'}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, with [n] citation markers:"
    )


def remember(deps: AgentDeps, question: str, answer_text: str) -> None:
    """Write the turn to long-term memory. Never fatal to the current answer."""
    try:
        deps.memory.add(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ],
            deps.user_id,
        )
    except Exception:
        pass


def blocked_answer(question: str, reason: str, validator: str = "") -> Answer:
    """Uniform shape for a guardrail rejection, whichever framework produced it."""
    detail = f"{reason} (validator: {validator})" if validator else reason
    return Answer(question=question, text="", blocked=True, block_reason=detail)


def assemble(
    question: str,
    raw_answer: str,
    hits: Sequence[Retrieved],
    citations: Sequence[Citation],
    memories: Sequence[MemoryRecord] = (),
) -> Answer:
    """Attach only the citations the model actually used, and return the Answer."""
    text = raw_answer.strip()
    return Answer(
        question=question,
        text=text,
        citations=prune_citations(text, citations),
        retrieved=list(hits),
        memories_used=[m.text for m in memories],
    )


def run_turn(deps: AgentDeps, question: str) -> Answer:
    """The reference retrieve -> reason -> answer turn, framework-free.

    Framework adapters may call this directly (when the framework adds nothing
    beyond orchestration) or re-implement the same steps in their own idiom.
    """
    guard_in = deps.guardrails.check_input(question)
    if not guard_in.allowed:
        return blocked_answer(question, guard_in.reason, guard_in.validator)

    memories = recall(deps, question)
    hits = retrieve(deps, question)
    context, citations = format_context(hits, deps.max_context_chars)
    prompt = build_prompt(question, context, memories)
    raw = deps.inference.complete(prompt, system=SYSTEM_PROMPT)

    guard_out = deps.guardrails.check_output(raw, context=context)
    if not guard_out.allowed:
        return blocked_answer(question, guard_out.reason, guard_out.validator)

    answer = assemble(question, guard_out.text or raw, hits, citations, memories)
    remember(deps, question, answer.text)
    return answer
