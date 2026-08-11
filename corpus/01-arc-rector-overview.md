# Arc Rector: A Complete Agentic RAG Stack From Open Source

Arc Rector is a reference implementation of a complete agentic retrieval-augmented
generation stack in which every layer is an open-source or self-hostable
component. The design goal is that a developer can clone the repository, run
`docker compose up`, and have a working agentic RAG system with no vendor
account, no credit card, and no API key.

## The nine levels

The stack is organised into nine levels. Each level is a swappable adapter behind
a single interface, and each has a default implementation that runs out of the
box.

Level 0 is inference and deployment: the component that turns a prompt into
text. The default is Ollama, which serves open-weights models locally over HTTP.

Level 1 is evaluation and observability. The default tracer is Langfuse,
self-hosted in the project's own compose file. The default evaluator is Ragas.

Level 2 is the language model itself. The default is Llama 3.1 8B, served through
Level 0.

Level 3 is the agent framework, which orchestrates the retrieve, reason, and
answer steps. The default is LangGraph.

Level 4 is the vector database. The default is Qdrant.

Level 5 covers embeddings and reranking. The default embedding model is Nomic
Embed, and reranking is disabled by default.

Level 6 is data ingestion and parsing. The default parser is Docling.

Level 7 is memory and context, which is what allows facts to persist between
conversation turns. The default is Mem0.

Level 8 is safety and guardrails, applied to both model input and model output.
The default is Guardrails AI.

## Why the level split matters

Most RAG tutorials hard-wire a single vendor path, which makes it impossible to
tell which component is responsible for a given behaviour. Arc Rector separates
the substance of a turn from its orchestration, so that changing the agent
framework does not change what gets retrieved, and changing the vector database
does not change how the answer is written.

Because of that separation, a user can run the same question against two
different vector databases and two different agent frameworks and directly
compare the results. If answer quality is unchanged when the framework changes,
then the framework is earning its place on ergonomics rather than on quality,
which is a useful thing to know before adopting one.

## Configuration

Every level is named in a single file called `config.yaml`. Changing one line
swaps that layer with no code edits. Every value can also be overridden by an
environment variable, so a single run can use a different stack without editing
any file at all.

## What Arc Rector is not

Arc Rector is a complete and correct starting point. It is not a production
system. It has no authentication, no multi-tenancy, no rate limiting, no backup
strategy, and no circuit breakers. Those are described in the project's
production hardening notes, and they must be addressed before the stack is
exposed to real traffic.
