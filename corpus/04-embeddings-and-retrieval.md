# Embeddings, Prefixes, and Retrieval Quality

The embedding layer is where retrieval quality is decided, and it is also where
the most silent failures occur, because a misconfigured embedding model does not
raise an error. It simply retrieves slightly worse documents forever.

## Nomic Embed

Nomic Embed is an Apache 2.0 licensed embedding model with 137 million
parameters and a context window of 8192 tokens. It produces 768-dimensional
vectors and outperforms OpenAI's text-embedding-ada-002 on the MTEB benchmark
while being a download of roughly 274 megabytes. Because it can be served through
Ollama over plain HTTP, it requires no Python machine learning stack at all,
which is why Arc Rector uses it as the default.

The critical detail is that Nomic Embed version 1 and version 1.5 are
instruction-prefixed models. Text being stored in the index must be embedded with
the prefix `search_document:` and a query must be embedded with the prefix
`search_query:`. Omitting these prefixes does not produce an error and does not
produce obviously broken output. It produces a measurable drop in retrieval
quality that is very difficult to notice without an evaluation harness, which
makes it one of the more expensive silent bugs in the field.

## Jina embeddings

Jina publishes open-weights embedding models, including small English models that
run comfortably on a CPU. They are a reasonable alternative to Nomic. One caution
is that Jina version 2 models require the `trust_remote_code` flag when loaded
through the sentence-transformers library, which means executing model-authored
code from the model repository. That is a supply-chain consideration worth making
explicitly rather than accepting by default.

## Dimensionality and swapping models

Vector collections are created with a fixed dimensionality. Nomic Embed produces
768 dimensions; the widely used all-MiniLM-L6-v2 model produces 384. Changing the
embedding model therefore requires re-ingesting the corpus, because the existing
vectors are not comparable with vectors from a different model. A system that
silently mixes embeddings from two models will return nonsense with full
confidence.

## Reranking

Retrieval with a bi-encoder embedding model is fast because document vectors are
computed once in advance, but it is approximate: the query and the document are
never seen together. A cross-encoder reranker takes the query and each candidate
document as a single joint input, which is far more accurate and far slower.

The standard pattern is therefore two-stage. Retrieve a wide candidate set,
perhaps twelve chunks, using the fast bi-encoder, then rerank those candidates
with a cross-encoder and keep the best four. This gives most of the accuracy of a
cross-encoder at a small fraction of its cost, because the expensive model only
ever sees a handful of documents.

## Chunking

Chunk boundaries matter as much as the model. A fact split across two chunks may
be retrievable from neither. The usual mitigation is overlap: carrying a small
amount of trailing context from one chunk into the next, so that a statement
spanning a boundary still appears intact in at least one chunk. Splitting on
paragraph boundaries rather than at a fixed character count also helps, because
paragraphs are already semantic units.
