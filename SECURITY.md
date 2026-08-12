# Security

## What this project is, in security terms

Arc Rector is a **reference implementation**, not a product. It is single-tenant
and meant to run on loopback on a machine you control. Authentication is one
optional shared password over the web UI (`ARC_UI_BASIC_AUTH_USER` /
`ARC_UI_BASIC_AUTH_PASSWORD`, off by default) — set it before any tunnel, and do
not mistake it for identity. `PRODUCTION.md` lists what is deliberately absent
and what to build before real traffic. Those absences are scope, not
vulnerabilities.

What *is* a vulnerability here: anything that harms someone who clones this repo
and follows the README.

## The committed credentials are not a leak

`docker-compose.yml` ships fixed, weak, deliberately public passwords —
`postgres/postgres`, `minio/miniosecret`, a known Langfuse key pair, a known
`ENCRYPTION_KEY`. That is intentional. It is what makes tracing work on first
boot with nothing to configure, on a stack whose every port is bound to
`127.0.0.1`.

They are safe for exactly one thing: a local stack on your own machine.

**They are not defaults you can deploy.** `docker-compose.a1.yml` — the profile
for a shared, tunnel-reachable box — declares every credential as a *required*
variable, so it refuses to start rather than boot on the values published here.
Generate real ones:

```bash
./deploy/a1-setup.sh --gen-secrets
```

`NVIDIA_API_KEY` is the only third-party key this project can use, it is
optional, `.env` is gitignored, and no adapter prints, logs, or echoes it.

## Reporting a vulnerability

Open a **private** security advisory:
<https://github.com/dev48v/arc-rector/security/advisories/new>

Please do not open a public issue for anything exploitable. A first response
should take a few days; this is a personal project, not a staffed one.

Useful things to include: the version or commit, which levels were selected
(`arc-rector levels`), and the smallest reproduction you can manage.

## In scope

- Remote code execution, SSRF, or path traversal reachable from a document
  source, a URL, or an API request
- Anything that makes the FastAPI server exploitable when run as documented
- Secrets escaping into logs, traces, API responses, or the repository
- Indirect prompt injection that defeats the L8 layer as configured
- A supply-chain problem in what this repo pins or ships

## Out of scope

These are documented absences, covered in `PRODUCTION.md`:

- No authentication, authorisation, or multi-tenancy
- No rate limiting, quota, or spend cap
- No HA, backups, or disaster recovery
- Regex guardrails being defeatable by paraphrase — stated in the code and the
  README; they are a speed bump beneath a model-based check, never a wall
- Anything that requires already having access to the host

## Hardening already applied

- Every published port binds to `127.0.0.1` in both compose files.
- Every image is pinned to an explicit version; no `:latest`.
- The UI container runs as uid 65534 and cannot write its own source tree.
- The L6 URL loader rejects non-`http(s)` schemes, credentials in the URL, and
  any host resolving to a private, loopback, link-local (cloud metadata),
  reserved, or multicast address. Redirects are followed by hand so every hop is
  re-checked, and the body is capped at 10 MB.
- Retrieved documents are fenced as untrusted data in the prompt, delimiter
  forgeries are stripped, and the guardrail layer screens context for embedded
  instructions.
- API errors return an id, not an exception message; the detail goes to the log.
- The server sets CSP, `X-Frame-Options`, `X-Content-Type-Options`,
  `Referrer-Policy`, and rejects cross-site requests to `/api/chat*`.
- The UI renders every model-controlled and document-controlled string with
  `textContent`; there is no `innerHTML` path.
