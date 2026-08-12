# Production readiness

## What Arc Rector is

A **complete, correct, zero-cost reference stack**. Every level of an agentic RAG
system is present, wired to a real implementation, and swappable. The retrieval
works, the citations resolve, the guardrails fire, the traces land, and the tests
pass. It is a good place to start from and a good place to learn from.

## What Arc Rector is not

**It is not production-ready, and it is not close.** It is a single-tenant demo
with committed credentials, no authentication, and no operational safety net.

That is a scope decision, not an oversight. Everything on this page is missing
*on purpose*, because building it would have buried the thing this repo exists to
show. What follows is the bill for that decision, itemised.

Do not put this in front of real traffic or real data until you have worked
through it.

---

## What is already done

Worth knowing before you start, so you do not rebuild it. These are the security
properties the repo ships with, and the tests that hold them in place:

| Property | Where |
|---|---|
| Every published port binds to `127.0.0.1`, both compose files | `docker-compose.yml`, `docker-compose.a1.yml` |
| Every image pinned to an explicit version, no `:latest` | both compose files, CI asserts it |
| Deploy profile **refuses to boot** on the repo's published passwords | `docker-compose.a1.yml` (`${VAR:?}`), `a1-setup.sh --gen-secrets` |
| URL ingestion rejects non-public addresses, bad schemes, oversized bodies | `l6_ingestion/plaintext.py`, `tests/test_ingestion.py` |
| Retrieved documents fenced as untrusted data; delimiter forgery stripped | `citations.py`, `tests/test_citations.py` |
| Guardrails screen retrieved context for embedded instructions | `l8_guardrails/builtin.py`, `rag_core.guard_context` |
| API errors return an id, never an exception message | `server.py`, `tests/test_server.py` |
| CSP, `X-Frame-Options`, `nosniff`, `Referrer-Policy`; cross-site `/api/chat*` rejected | `server.py` |
| UI renders every model- and document-controlled string via `textContent` | `static/index.html` |
| The UI container runs as uid 65534 and cannot write its source tree | `deploy/Dockerfile.ui` |

Everything below is what those do **not** cover.

---

## 1. Authentication and multi-tenancy

**Today:** one optional door lock and nothing else. Setting
`ARC_UI_BASIC_AUTH_USER` and `ARC_UI_BASIC_AUTH_PASSWORD` puts HTTP basic auth
over every UI and API route, which is enough to stop a public tunnel being an
open invitation to spend your CPU. It is off by default, it is a single shared
credential, and it is not identity: it cannot tell two users apart, so none of
what follows is solved by it.

Every other service in the compose file is unauthenticated, and `user_id` is a
string the caller supplies. The web UI is the sharpest edge: `session_id` in
`POST /api/chat` becomes the L7 memory `user_id` after nothing but character
sanitising, so anyone who guesses another browser's session id reads its
remembered facts. It is bound to `127.0.0.1` for exactly that reason.

**Before real traffic:**

- Put an authenticating gateway in front of everything, the UI included.
  Cloudflare Access in front of a named tunnel is the cheapest version of this
  and takes an afternoon; `deploy/tunnel.sh` already points at it.
- Derive `user_id` from a verified session token, **never** from request input —
  which is what `session_id` is today. Delete the `session_id` field when you do;
  leaving it accepted-but-ignored is how it comes back.
- Make tenancy structural, not conventional: a per-tenant Qdrant collection, or a
  mandatory tenant filter applied inside the store adapter. A filter applied at
  the call site will eventually be omitted at one call site.
- Corpus documents need their own access control. Retrieval is an
  information-disclosure channel: if a user can ask questions, they can extract
  any chunk the retriever can reach, regardless of what your UI displays.
- `/api/health` and `/api/config` are diagnostic surfaces that name your internal
  hostnames and adapter settings. Authenticate them or drop them in production.

**Done when:** an unauthenticated request to every port returns 401, and a test
proves user A cannot read user B's memories by changing one request field.

## 2. Secrets management

**Today:** `docker-compose.yml` contains fixed, weak, deliberately committed
passwords — `postgres/postgres`, `minio/miniosecret`, a hardcoded
`ENCRYPTION_KEY`, and Langfuse keys seeded through `LANGFUSE_INIT_*`. That is
intentional and safe for exactly one thing: a stack listening on loopback on your
own machine. It is what makes tracing work on first boot with nothing to
configure.

`docker-compose.a1.yml` — the shared-box profile — no longer has that escape
hatch. Every credential is a required variable, so it fails to start rather than
quietly booting on values that are published in a public repository.

**Before real traffic:**

- `./deploy/a1-setup.sh --gen-secrets`, then back the `.env` up somewhere you
  will still have it after the box dies. Losing `LANGFUSE_ENCRYPTION_KEY` makes
  every stored trace permanently unreadable.
- Move them out of `.env` into a real secret store — Docker/Kubernetes secrets,
  Vault, SSM Parameter Store — and rotate on a schedule. `.env` on disk is one
  container escape away from being the whole estate.
- `ENCRYPTION_KEY` must be `openssl rand -hex 32` and must be **quoted** in YAML.
  Unquoted, an all-digit hex string parses as the integer `0` and Langfuse
  rejects it at boot with a message that does not mention YAML.
- Audit your history before making a fork public. A rotated secret still sitting
  in a commit is still a leaked secret.

**Done when:** no credential appears in any tracked file, and rotating one is a
documented procedure somebody other than you has run.

## 3. Rate limits and cost caps

**Today:** none. A single caller can issue unbounded queries, and each query is
an unbounded generation. The only controls are `max_input_chars` on L8 and the
server's one-turn-at-a-time lock.

**Before real traffic:**

- Per-user and per-IP rate limits at the gateway, not in application code. Code
  that limits itself is code an attacker can make skip the limit.
- Cap `num_predict` and `num_ctx` on the L0 adapter — both are already settings.
  Unbounded generation is unbounded compute.
- Keep `max_input_chars` low. It is a denial-of-wallet control, not a formatting
  nicety.
- If you move L0 to a paid API, set a hard spend cap **at the provider**. Your
  own code is the thing that will be broken when it matters.
- Queue and shed load. Ollama serialises generation: under concurrency, request N
  waits for all N−1 before it. Two callers on one Ollama is already a degraded
  system. The server's lock bounds the damage and does nothing to bound the
  queue — a public UI needs an explicit queue with a rejection path and a visible
  "we are full" response.
- Cap concurrent SSE connections. Each holds a thread and a queue for the length
  of a generation.

**Done when:** a load test at 10× expected traffic produces rejections rather
than a growing queue, and the bill has a ceiling you did not have to enforce.

## 4. Retries, timeouts, circuit breakers

**Today:** every adapter sets an HTTP timeout, and the Mem0 adapter has a
wall-clock watchdog that falls back to local memory. That is the extent of it.
No retries, no breakers.

**Before real traffic:**

- Retry with exponential backoff and jitter on transient failures only. Never
  blindly retry a non-idempotent write.
- A circuit breaker per dependency. When the vector store is down, fail fast; do
  not queue thousands of requests that will each wait out a full timeout.
- Decide explicitly, per layer, whether failure is fatal or degraded. This
  repo's stance: memory and tracing degrade silently, guardrails and retrieval do
  not. Write your stance down — an undocumented policy becomes an accident.
- A guardrail that cannot run must **fail closed**. An input check that silently
  passes when the classifier is unreachable is worse than no guardrail, because
  it is trusted. Note that `fallback_to_builtin: true` in `config.yaml` is a
  fail-*open* toward a weaker check; that is a deliberate demo default and the
  wrong one for production.

**Done when:** killing any single container degrades the system in a way you
predicted in writing beforehand.

## 5. Durability and backups

**Today:** named Docker volumes. `docker compose down -v` destroys them, and
nothing else protects them.

**Before real traffic:**

- Qdrant's snapshot API on a schedule, shipped off-host. Test the restore — an
  untested backup is a hypothesis.
- Keep the **source corpus** as the real source of truth and make re-ingestion a
  routine automated operation. Vectors are derived data; if you can always
  rebuild them, an outage is an inconvenience rather than a loss.
- Version your embedding model alongside the collection. Changing L5 changes
  dimensionality, and mixing vectors from two models returns confident nonsense
  with no error. `ensure_collection` raises on a mismatch — do not remove that
  check.
- Langfuse's Postgres and ClickHouse need their own backup policy if traces are
  an audit record rather than debug output.
- State your RTO and RPO as numbers. "We have backups" is not a recovery plan.

**Done when:** you have restored from a backup into a clean host, timed it, and
the number is inside your stated RTO.

## 6. Scaling, health, and observability of the system itself

**Today:** one container each, `restart: unless-stopped`, health checks on the
infrastructure containers, and a single Ollama serialising every generation. The
A1 profile sets a `mem_limit` on every service; the local one does not.

**Before real traffic:**

- Real liveness and readiness endpoints for your own service. `/api/health`
  probes dependencies, which makes it a readiness check, not a liveness one —
  wire them separately or a slow Qdrant will get your UI killed and restarted.
- Scale the model layer horizontally behind a load balancer, or move to vLLM,
  which is built for batched concurrent serving in a way Ollama is not.
- Memory limits on every container in every profile. An unbounded ClickHouse will
  take the box down and your RAG outage will look like an unrelated failure.
- Watch p99, not the mean. RAG latency is dominated by the tail: a cold model
  load or a large context is seconds, not milliseconds.
- Alert on **retrieval quality**, not just uptime. A system returning irrelevant
  chunks is 100% available and completely broken. Track the rate of "I don't know
  based on the provided documents" and the rate of answers citing nothing; both
  move before users complain.
- Structured logs with a request id. The server logs an error id on failure —
  make sure something is actually collecting those, or the id points at nothing.

**Done when:** you can answer "was retrieval worse this week than last week?"
from a dashboard rather than by asking someone.

## 7. Prompt-injection hardening

**Today:** retrieved documents are fenced as untrusted data, the system prompt
names those fences and says the content inside is data rather than instructions,
delimiter forgery is stripped, and the L8 layer scans context for
instruction-shaped text and warns the model when it finds any.

That is a real improvement over filtering input alone. It is still **not a
solution**, and no prompt-level defence is. A sufficiently persuasive document
can still talk a model out of its instructions.

**Before real traffic:**

- Layer a model-based classifier over the patterns — the `llamaguard` and `nemo`
  adapters exist for this. Regex plus a classifier is defence in depth; either
  alone is not.
- Sanitise at **ingestion**, not only at query time. That is where injected
  instructions actually enter the system, and it is the only point where you can
  reject a document instead of arguing with it later.
- Constrain the blast radius. If the agent gains tools, give it least privilege
  and require confirmation for side effects. An injection that can only produce
  bad text is an incident; one that can call an API is a breach.
- Never interpolate retrieved text into anything executable — SQL, shell, or
  code — and never into an outbound URL.
- Curate what you ingest. An untrusted corpus is an untrusted system, and no
  amount of prompt engineering changes that.

**Done when:** you have a red-team corpus of poisoned documents in CI, and it
fails the build when a new prompt or model regresses against it.

## 8. Server-side request forgery, at the edges

**Today:** the L6 plaintext loader validates every URL and every redirect hop
against a public-address rule, so the obvious attack — pointing ingestion at
`169.254.169.254` and reading cloud instance metadata — is closed, along with
non-`http(s)` schemes, credentials in the URL, and unbounded downloads.

**Before real traffic:**

- The check resolves DNS, then `requests` resolves it again to connect. A name
  that changes answers between those two moments (**DNS rebinding**) still gets
  through. Closing it properly needs a transport that connects to the IP the
  check validated. Do this if you accept URLs from users rather than operators.
- `docling` (the default), `unstructured` and `scrapy` fetch through their own
  machinery, so each calls the gate explicitly before handing a URL over; a test
  asserts they still do. **`firecrawl` is the exception** — it is a hosted API
  that fetches from Firecrawl's infrastructure, not yours, so this gate does not
  and cannot apply. `scrapy.crawl()` follows discovered links, and only the start
  URL is checked; do not point it at untrusted input.
- Better still, put egress behind an allowlist proxy and give the container no
  other route out. A network-level control does not have to be right about DNS.

**Done when:** the ingestion container cannot open a connection to anything that
is not on an explicit allowlist.

## 9. Deploy-gating evaluations

**Today:** `eval_harness.py` runs eight pairs, manually. CI runs the unit tests
and a dependency audit, not the evals.

**Before real traffic:**

- Run evals in CI on every change to a prompt, a chunking parameter, a model, or
  an embedding model. All four change retrieval quality invisibly; none of them
  will fail a unit test.
- Set thresholds and **fail the build**. An eval you can ignore is documentation,
  not a gate.
- Grow the golden set from real production failures. Every incident becomes a
  permanent test case.
- Track scores over time. Absolute values from a small local judge are
  directional; the **delta between runs** is the signal.
- Keep an offline deterministic evaluator (this repo's `builtin`) so CI is never
  gated on a model being reachable.

**Done when:** a pull request that quietly worsens retrieval cannot be merged.

## 10. Supply chain

**Today:** every image is pinned to an explicit version and CI runs `pip-audit`
weekly and on every change. `pip-audit` reports without failing the build,
because most findings are transitive and unfixable from this repo.

**Before real traffic:**

- Pin images by **digest**, not tag. A tag is a mutable pointer; `@sha256:...` is
  not. Then run something that proposes digest bumps for you.
- Lock Python dependencies (`pip-compile`, `uv lock`) and install from the lock
  in CI and in the image. Floors in `pyproject.toml` describe what works, not
  what shipped.
- Make the audit fail the build once you actually control your transitive tree.
  A red build nobody can turn green is a build people learn to ignore.
- Generate an SBOM per release and keep it with the artifact.

**Done when:** you can name every version running in production from an artifact,
without connecting to the box.

## 11. Privacy, retention, and compliance

**Today:** Langfuse records full prompts and completions, which include whatever
your users typed. Memory persists user facts indefinitely with no expiry and no
deletion path beyond `reset()`.

**Before real traffic:**

- Decide what may be traced. Redact PII before it reaches the tracer, or disable
  input/output capture on sensitive paths. The tracer is the largest unplanned
  collection of user text in this stack.
- Set retention on traces, memory, and the vector store. "Forever" is a decision,
  and usually the wrong one.
- Implement deletion properly. A user's right to erasure covers their memories,
  their traces, and any chunks derived from their documents — three stores, not
  one.
- Confirm the licence of every model you ship commercially. Llama models are
  **open weights, not open source**; see the README's licence caution and
  `corpus/03-open-weights-vs-open-source.md`.
- Check the licence of everything in your corpus too. Ingesting a document does
  not grant you the right to serve it back.

**Done when:** a deletion request is a command you can run, and you can say how
long any given record lives without looking it up.

---

## A realistic hardening order

Do these in order. Each one makes the next one cheaper.

1. **Lock the network down and authenticate everything.** Nothing else on this
   list matters while the answer to "who can reach it?" is "anyone".
2. **Generate real credentials and move them to a secret store.**
   `./deploy/a1-setup.sh --gen-secrets` is step one of two; the secret store is
   step two.
3. **Rate limits and a hard spend cap at the provider.**
4. **Backups, and a restore you have actually performed.**
5. **Evals in CI with thresholds that can fail a build**, plus a poisoned-document
   red-team set.
6. **A model-based guardrail layer**, and egress restricted to an allowlist.
7. **Health checks, memory limits everywhere, tail-latency and retrieval-quality
   alerting.**
8. **Retention and deletion policies**, implemented rather than written down.

Only after all eight does "self-hosted and free to run" also mean "safe to run".
