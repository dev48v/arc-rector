# Choosing an Open-Source Vector Database

Arc Rector ships working adapters for four open-source vector databases. They are
genuinely interchangeable at the interface level, but they differ in operational
character, and the differences matter more than the benchmark numbers.

## Qdrant

Qdrant is written in Rust and released under the Apache 2.0 licence. It is the
default in Arc Rector because it runs as a single container of roughly 275
megabytes with no external dependencies, it supports cosine distance natively,
and its payload model allows the entire text chunk to be stored alongside the
vector. That last property means retrieval requires exactly one network round
trip rather than a vector search followed by a separate document fetch.

The main operational surprise with Qdrant is its point identifier rule. A point
ID must be either an unsigned integer or a UUID. An arbitrary hexadecimal string
is rejected. Systems that use content hashes as chunk identifiers must therefore
reinterpret those hashes as UUIDs before writing.

## Chroma

Chroma is released under Apache 2.0 and is the easiest of the four to start with,
because it can run entirely in-process with a persistent local directory and no
server at all. This makes it excellent for notebooks, tests, and small local
corpora.

Two things catch people out. First, Chroma returns cosine *distance*, not
similarity, so a score must be converted with one minus distance before it can be
compared against other stores. Second, Chroma metadata values must be scalars;
nested structures and null values are rejected and must be encoded or dropped.

## pgvector

pgvector is a PostgreSQL extension under the PostgreSQL licence. Its argument is
operational rather than technical: if an organisation already runs PostgreSQL,
pgvector means no new database to back up, secure, monitor, or explain to a
platform team. Vectors live in the same transaction as the rest of the
application data, which makes consistency straightforward.

The important limit is dimensionality. The indexed pgvector type supports at most
2000 dimensions for both HNSW and IVFFlat indexes. Above that, search still works
but falls back to a sequential scan. A second subtlety is that queries must order
by the raw distance operator ascending in order for the index to be used;
ordering by a converted similarity score descending is mathematically identical
but silently degrades every search into a full table scan.

## Milvus

Milvus is an Apache 2.0 distributed vector database aimed at billion-scale
workloads. It is the most capable of the four and the heaviest: a full deployment
involves etcd, object storage, and multiple coordinator services.

Milvus has two behaviours worth knowing. Its default consistency level is bounded
staleness, which means a chunk that was just written may briefly be invisible to
search, breaking naive ingest-then-query demonstrations unless strong consistency
is requested. And a collection that exists is not necessarily loaded; after a
server restart, collections must be explicitly loaded before they will accept
searches.

## How to choose

For a local demo or a single-node service, Qdrant is the safe default. For tests
and notebooks, Chroma removes the container entirely. For a team that already
operates PostgreSQL, pgvector is usually the correct answer despite being the
least specialised. Milvus is justified when the corpus genuinely reaches hundreds
of millions of vectors, and rarely before then.
