# Production readiness

## What Arc Rector is

A **complete, correct, zero-cost reference stack**. Every level of an agentic RAG system is present, wired to a real implementation, and swappable. The retrieval works, the citations resolve, the guardrails fire, the traces land, and the tests pass. It is a good place to start from and a good place to learn from.

## What Arc Rector is not

**It is not production-ready, and it is not close.** It is a single-tenant demo with committed credentials, no authentication, and no operational safety net. Nothing below is a nice-to-have; each item is something that will bite you specifically, in a way that is hard to debug after the fact.

Do not put this in front of real traffic or real data until you have worked through this list.

---

## 1. Authentication and multi-tenancy

**Today:** none. Every API in the compose file is unauthenticated, `user_id` is a string the caller supplies, and any caller can pass any `user_id` and read another user's memories.

**Before real traffic:**
- Put an authenticating gateway in front of everything. Qdrant, Langfuse, MinIO and Postgres are bound to `127.0.0.1` in the compose file precisely because none of them should ever be directly reachable.
- Derive `user_id` from a verified session token, **never** from request input.
- Make tenancy structural, not conventional: a per-tenant Qdrant collection, or a mandatory tenant filter applied in the store adapter itself so no caller can forget it. A filter applied at the call site will eventually be omitted at one call site.
- Corpus documents need their own access control. Retrieval is an information-disclosure channel: if a user can ask questions, they can extract any chunk the retriever can reach, regardless of what your UI shows.

## 2. Secrets management

**Today:** `docker-compose.yml` contains fixed, weak, deliberately committed passwords — `postgres/postgres`, `minio/miniosecret`, a hardcoded `ENCRYPTION_KEY`, and Langfuse API keys seeded through `LANGFUSE_INIT_*`. This is intentional: it makes tracing work on first boot with nothing to configure, on a stack listening only on localhost.

**Before real traffic:**
- Regenerate every one of them. `ENCRYPTION_KEY` must be `openssl rand -hex 32`, and must be **quoted** in YAML — unquoted, an all-digit hex string is parsed as the integer `0` and Langfuse rejects it at boot.
- Move them to a real secret store (Docker/Kubernetes secrets, Vault, SSM Parameter Store). Rotate on a schedule.
- Keep `.env` gitignored. `NVIDIA_API_KEY` is the only key this project can use, and no adapter prints, logs, or echoes it.
- Audit your history before making anything public: a rotated secret that is still in a commit is still a leaked secret.

## 3. Rate limits and cost caps

**Today:** none. A single caller can issue unbounded queries, and each query is an unbounded generation.

**Before real traffic:**
- Per-user and per-IP rate limits at the gateway.
- Cap `num_predict` and `num_ctx` on the L0 adapter (both are settings already). Unbounded generation is unbounded compute.
- The `max_input_chars` guardrail is a denial-of-wallet control, not a formatting nicety. Keep it low.
- If you move L0 to a paid API, set a hard spend cap **at the provider**, not in your own code. Your code is what will be broken when it matters.
- Queue and shed load. Ollama serialises requests: under concurrency, request N waits for all N-1 before it. Two callers on one Ollama instance is already a degraded system.

## 4. Retries, timeouts, circuit breakers

**Today:** every adapter sets an HTTP timeout, and the Mem0 adapter has a wall-clock watchdog that falls back to local memory. That is the extent of it. There are no retries and no breakers.

**Before real traffic:**
- Retry with exponential backoff and jitter on transient failures only. Never blindly retry a non-idempotent write.
- Add a circuit breaker per dependency. When the vector store is down, fail fast; do not queue thousands of requests that will each wait for a full timeout.
- Decide explicitly, per layer, whether failure is fatal or degraded. This repo's stance: memory and tracing degrade silently, guardrails and retrieval do not. Write your stance down — an undocumented policy becomes an accident.
- A guardrail that cannot run must **fail closed**. An input check that silently passes when the classifier is unreachable is worse than no guardrail, because it is trusted.

## 5. Vector store durability and backups

**Today:** a named Docker volume. `docker compose down -v` destroys it, and nothing else protects it.

**Before real traffic:**
- Use Qdrant's snapshot API on a schedule; ship snapshots off-host. Test the restore — an untested backup is a hypothesis.
- Keep the **source corpus** as the real source of truth and make re-ingestion a routine, automated operation. Vectors are derived data; if you can always rebuild them, an outage is an inconvenience rather than a loss.
- Version your embedding model alongside the collection. Changing L5 changes dimensionality, and mixing vectors from two models returns confident nonsense with no error. `ensure_collection` raises on a mismatch — do not remove that check.
- Langfuse's Postgres and ClickHouse need their own backup policy if traces are an audit record rather than debug output.

## 6. Scaling and health

**Today:** one container each, `restart: unless-stopped`, one health check on Qdrant, and a single Ollama serialising every generation.

**Before real traffic:**
- Real liveness and readiness endpoints for your own service. Readiness must check the vector store and the model, or you will route traffic to a pod that cannot answer.
- Scale the model layer horizontally behind a load balancer, or move to vLLM, which is built for batched concurrent serving in a way Ollama is not.
- Set memory limits on every container. An unbounded ClickHouse will take the box down and your RAG outage will look like an unrelated failure.
- Watch p99, not the mean. RAG latency is dominated by the tail: a cold model load or a large context is seconds, not milliseconds.
- Alert on retrieval quality, not just uptime. A system returning irrelevant chunks is 100% available and completely broken.

## 7. Prompt-injection hardening

**Today:** regex patterns over input and output. This is a speed bump. Say so out loud.

The threat that matters in RAG is **indirect** injection: instructions hidden in an ingested document, which the retriever faithfully places in your context window. Input filtering does nothing about it, because the attack never appears in the user's message.

**Before real traffic:**
- Treat retrieved content as **untrusted data**, never as instructions. Mark context boundaries explicitly and instruct the model that content inside them is quoted material.
- Layer a model-based classifier over the patterns — the `llamaguard` and `nemo` adapters exist for this. Regex plus a classifier is defence in depth; either alone is not.
- Sanitise at **ingestion**, not only at query time. That is where injected instructions actually enter the system.
- Constrain the blast radius: if the agent gains tools, give it least privilege and require confirmation for side effects. An injection that can only produce bad text is an incident; one that can call an API is a breach.
- Never interpolate retrieved text into anything executable — SQL, shell, or code.

## 8. Deploy-gating evaluations

**Today:** `eval_harness.py` runs eight pairs manually.

**Before real traffic:**
- Run evals in CI on every change to a prompt, a chunking parameter, a model, or an embedding model. All four change retrieval quality invisibly; none of them will fail a unit test.
- Set thresholds and **fail the build**. An eval you can ignore is documentation, not a gate.
- Grow the golden set from real production failures. Every incident should become a permanent test case.
- Track scores over time. Absolute values from a small local judge are directional; the **delta between runs** is the signal.
- Keep an offline deterministic evaluator (this repo's `builtin`) so CI is not gated on a model being reachable.

## 9. Privacy, retention, and compliance

**Today:** Langfuse records full prompts and completions, which include whatever your users typed. Memory persists user facts indefinitely with no expiry and no deletion path beyond `reset()`.

**Before real traffic:**
- Decide what may be traced. Redact PII before it reaches the tracer, or disable input/output capture on sensitive paths.
- Set retention on traces, memory, and the vector store. "Forever" is a decision, and usually the wrong one.
- Implement deletion properly. A user's right to erasure covers their memories, their traces, and any chunks derived from their documents.
- Confirm the licence of every model you ship commercially. Llama models are **open weights, not open source** — see the README's licence caution.

---

## A realistic hardening order

1. Lock the network down and put authentication in front of everything.
2. Regenerate every credential and move them into a secret store.
3. Add rate limits and hard cost caps.
4. Add backups, and test a restore.
5. Wire evals into CI with thresholds that can fail a build.
6. Add a model-based guardrail layer and treat retrieved text as untrusted.
7. Add health checks, memory limits, and tail-latency alerting.
8. Set retention and deletion policies.

Only after all eight does "self-hosted and free to run" also mean "safe to run".
