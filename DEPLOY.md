# Deploying Arc Rector for free

The whole point of this project is that it costs nothing to run. That should include the hosting. This document covers the one free tier that can actually run it, why the others cannot, and how to deploy without breaking anything already on the box.

> ⚠️ **The reference target is a SHARED machine.** It already runs a Hermes agent and a WhatsApp-presence tracker. Nothing in this document stops, prunes, or reconfigures anything that was already there, and neither should you. See [Shared-box rules](#shared-box-rules).

---

## Why Oracle Cloud Always Free is the only free host that fits

Arc Rector needs Docker, several GB of RAM, and a process that stays running. That combination rules out almost every free tier.

| Host | Free tier | Can it run Arc Rector? |
|---|---|---|
| **Oracle Cloud Always Free (A1.Flex)** | Up to 4 ARM OCPU + 24 GB RAM, 200 GB block storage, always on | ✅ **Yes.** The only free tier with enough RAM and a real always-on VM. |
| Vercel | Serverless functions | ❌ No Docker, no persistent process, execution time limits. Cannot host Qdrant or ClickHouse. |
| Cloudflare Pages / Workers | Edge runtime | ❌ No Docker, no filesystem, no long-running process. |
| Render free tier | 512 MB web service | ❌ Spins down when idle; 512 MB will not hold ClickHouse, let alone a model. |
| Fly.io free allowance | Small shared-CPU VMs | ⚠️ Docker yes, but the free RAM allowance is well under what the Langfuse stack needs. |
| GitHub Codespaces | 60 core-hours/month | ⚠️ Fine for a demo, but it is a dev environment, not a host — it stops. |
| Hugging Face Spaces | 16 GB free CPU Space | ⚠️ Good for a single app; not a multi-container stack with persistent volumes. |

Oracle's A1.Flex is genuinely unusual: an always-on ARM VM with real RAM and real block storage, at no cost, indefinitely.

### The constraints that come with it

- **ARM64 (aarch64).** Every image must have a `linux/arm64` variant. An accidental amd64 pull either fails outright or runs under emulation at unusable speed.
- **No GPU.** Inference is CPU-only. This is the binding constraint, and it is why the A1 profile defaults to a 3B model.
- **2 OCPU on the reference box.** Ollama serialises requests; concurrency is not free.
- **Ingress hardened to TCP 22 only.** No opened ports, no security-list edits. Public access is via a Cloudflare tunnel.
- **Oracle reclaims idle Always Free instances.** An instance averaging **under 20% CPU, network, and memory utilisation over a 7-day window** may be reclaimed. This workload — Langfuse ingesting, ClickHouse merging, Ollama resident — comfortably clears that bar. A box running *only* this stack idle is a box you might lose, so do not treat "it's free" as "it's permanent": keep the corpus in git and treat vectors as rebuildable.

---

## Measured headroom on the reference box

Taken from the live machine, not estimated:

| | |
|---|---|
| RAM total | 11 GB |
| RAM genuinely in use | ~1 GB (another ~10 GB is reclaimable page cache) |
| **RAM available** | **~10 GB** |
| Disk volume | 48 GB, 14 GB used |
| **Disk free** | **35 GB** |
| Already running | `hermes-0239ec4a`, `hermes-48224850`, `wa-presence-app-1`, `wa-presence-db-1` (MySQL), `wa-presence-phpmyadmin-1`, `wa-presence-cloudflared-1` |
| Heaviest neighbours | mysqld ~482 MB RSS, hermes ~224 MB, node ~136 MB, dockerd ~103 MB |

**Budget: ≤ 8 GB for Arc Rector**, leaving over 2 GB for the existing services plus headroom. `docker-compose.a1.yml` sets an explicit `mem_limit` on every service, totalling **7.9 GB**:

| Service | Limit | |
|---|---|---|
| ollama | 4.0 GB | the only one that really needs it |
| clickhouse | 2.0 GB | also capped in `deploy/clickhouse-low-mem.xml` |
| langfuse-web | 1.5 GB | |
| qdrant | 1.0 GB | |
| langfuse-worker | 0.9 GB | |
| postgres | 0.4 GB | |
| minio | 0.4 GB | |
| **ui** | **0.4 GB** | one Python process; capped so it can never be what OOMs the box |
| redis | 0.3 GB | plus `--maxmemory 200mb` |

A cgroup limit alone does not restrain ClickHouse — it sizes its caches from *host* RAM and will plan a query that exceeds the cgroup, then get OOM-killed. `deploy/clickhouse-low-mem.xml` tells it what it is actually allowed. That file is not optional.

### On disk, not on speed

35 GB free means model size is not the constraint. A 3B is ~2 GB, an 8B ~5 GB; both fit easily. **The default is 3B because the box is CPU-only ARM** — an 8B answers at roughly 2–5 tokens/second there, which is a bad demo. You can switch to 8B knowing that trade:

```bash
ARC_MODEL=llama3.1:8b ./deploy/a1-setup.sh
```

---

## Deploy

```bash
# on the VM
git clone https://github.com/dev48v/arc-rector && cd arc-rector
chmod +x deploy/*.sh

./deploy/a1-setup.sh
```

`a1-setup.sh` is idempotent and safe to re-run. In order it:

1. **Measures headroom first and aborts before touching anything** if available RAM is under 3 GB or free disk under 10 GB, printing the measured values either way.
2. Checks the architecture is aarch64 and that Docker is usable.
3. **Lists every running container that is not ours**, so you see what you are sharing with.
4. Refuses to start if 6333, 3000, 8800 or 11434 is held by a foreign process (it recognises its own containers, so re-running is fine).
5. Pulls arm64 images, builds the UI image, starts the stack, waits for real health endpoints — including `GET /api/config` on the UI, which only answers once the whole stack has been resolved.
6. Pulls `nomic-embed-text` and `llama3.2:3b`, skipping either if already present.
7. Prints per-container memory use and remaining host RAM.

```bash
./deploy/a1-setup.sh --status      # report only, changes nothing
./deploy/a1-setup.sh --pull-only   # pull images, do not start
```

### No model weights on the box at all

To skip Ollama entirely and use the NVIDIA NIM free tier as L0 — still no bill, and ~4 GB of RAM back:

```bash
echo 'NVIDIA_API_KEY=nvapi-...' > .env      # .env is gitignored
docker compose -f docker-compose.a1.yml up -d --scale ollama=0
export ARC_L0_INFERENCE=nim
```

Embeddings then need a path too: either keep Ollama for `nomic-embed-text` only (small and fast on CPU), or switch L5 to `sentence-transformers`.

---

## Going public: Cloudflare tunnel

The box exposes **TCP 22 and nothing else**. Do not change that. `cloudflared` makes an *outbound* connection and Cloudflare proxies traffic back down it, so nothing needs to be opened.

**The tunnel points at the web UI on `:8800`.** That is the thing a person is meant to look at — a chat pane, the nine active levels, and the retrieval detail behind every answer. Langfuse on `:3000` is the second tunnel you might want, and only if you intend to click through to traces.

```bash
./deploy/tunnel.sh 8800       # the UI  <- start here
./deploy/tunnel.sh 3000       # Langfuse, if you want the trace links to open
./deploy/tunnel.sh 6333       # the Qdrant dashboard
```

Every port here is bound to `127.0.0.1` in `docker-compose.a1.yml`, including the UI. The tunnel reaches them from inside the machine; none of them is listening on a public interface.

### Making the trace links work through the tunnel

Inside the compose network the UI talks to Langfuse at `http://langfuse-web:3000`, which no browser can resolve. Every "Langfuse trace ↗" link on the page would 404. Set the address a visitor actually reaches Langfuse on:

```bash
echo 'ARC_UI_TRACE_BASE=https://langfuse.example.com' >> .env   # .env is gitignored
docker compose -f docker-compose.a1.yml up -d ui
```

Without a Langfuse tunnel, leave it unset and the links point at `localhost:3000` — correct if you are on the box or port-forwarding over SSH, broken for anyone else. Better to have an obviously-local link than a silently wrong one.

### What the UI container can and cannot do

The image is built with `EXTRAS=ui` on ARM, which keeps it small and its build short. Two consequences, both visible on the page rather than hidden:

- **L7 and L8 run their dependency-free adapters** (`local`, `builtin`), pinned by env in the compose file. The sidebar says `local` and `builtin`, because that is what is running. Build with `--build-arg EXTRAS=ui,memory,guardrails` and drop the two env pins to get the `config.yaml` defaults.
- **It cannot ingest.** Docling pulls torch and layout models; a query-only service has no use for them. Ingest from the CLI on the box (`pip install -e ".[ingest]" && arc-rector ingest --reset`) or from your laptop against the same Qdrant. The vectors are what the UI reads, and they outlive the container.

**Quick tunnel** gives an ephemeral `*.trycloudflare.com` URL. No account, no DNS, no config. The URL changes on every restart, the tunnel dies with the process, and **it is unauthenticated** — anyone with the link reaches the service. Demos only. Never point one at real data.

**Named tunnel** gives a stable hostname on a domain you control, survives restarts, and can sit behind Cloudflare Access for real authentication — which is what you want for anything beyond a demo, because Arc Rector has no auth of its own.

```bash
cloudflared tunnel login
cloudflared tunnel create arc-rector
cloudflared tunnel route dns arc-rector arc.example.com
./deploy/tunnel.sh 8800 named arc-rector arc.example.com
```

`tunnel.sh` writes its **own** config at `~/.cloudflared/arc-rector-config.yml` and runs its **own** process with its own pidfile and log. It detects the existing `wa-presence-cloudflared-1` container, warns that it is there, and leaves it completely alone.

---

## Shared-box rules

The reference VM runs other people's work. Breaking it is easy and unnecessary.

**Never run any of these:**

```bash
docker system prune -a        # deletes images and volumes for EVERY project
docker compose down           # without -f, acts on whatever compose file is nearby
docker stop $(docker ps -q)   # stops the Hermes agent and wa-presence too
docker volume prune
```

**Always scope to this project:**

```bash
docker compose -f docker-compose.a1.yml ps
docker compose -f docker-compose.a1.yml logs -f langfuse-web
docker compose -f docker-compose.a1.yml restart qdrant
```

Every container here is named `arc-rector-*` so it can never be confused with `hermes-*` or `wa-presence-*`. If a command would affect a container whose name does not start with `arc-rector-`, it is the wrong command.

### Checking headroom

```bash
free -m                                          # look at "available", not "free"
df -h /
docker stats --no-stream
./deploy/a1-setup.sh --status
```

On Linux most of "used" memory is reclaimable page cache. **`available` is the number that matters** — 1 GB "free" alongside 10 GB "available" is a healthy box, not a full one.

---

## Rolling back and tearing down

```bash
# stop Arc Rector, keep all data
docker compose -f docker-compose.a1.yml down

# stop and DELETE every Arc Rector volume (vectors, traces, models)
docker compose -f docker-compose.a1.yml down -v

# stop one service only
docker compose -f docker-compose.a1.yml stop langfuse-web

# stop the tunnel
kill $(cat /tmp/arc-rector-tunnel.pid)
```

`down -v` removes only volumes belonging to the `arc-rector` compose project. Other projects' volumes are untouched — which is exactly why the project name is pinned in the compose file rather than inferred from the directory.

Re-deploying after a teardown is `./deploy/a1-setup.sh` again. The corpus lives in git and the vectors are derived data, so a full rebuild is an ingest, not a restore.

---

## Troubleshooting

**`no matching manifest for linux/arm64`** — an image without an ARM build. Every image in `docker-compose.a1.yml` was chosen for arm64 support; if you added one, check its tags.

**ClickHouse restart-loops** — almost always memory. Confirm `deploy/clickhouse-low-mem.xml` is actually mounted (`docker compose -f docker-compose.a1.yml config | grep low-mem`) and check `docker inspect arc-rector-clickhouse | grep OOMKilled`.

**Langfuse exits at boot with a Zod error about `ENCRYPTION_KEY`** — the key must be **quoted** in YAML. Unquoted, an all-digit hex string parses as the integer `0`.

**Generation is painfully slow** — expected on CPU-only ARM. Use the 3B, lower `ARC_PIPELINE__TOP_K` and `ARC_L0_INFERENCE__NUM_CTX`, set `ARC_L0_INFERENCE__NUM_THREAD` to the core count, or move L0 to NIM.

**`a1-setup.sh` aborts on preflight** — that is the script working. Free memory or disk; do not lower the thresholds to get past it.

**The tunnel URL 404s** — the tunnel is up but nothing is listening on that port. Check `./deploy/a1-setup.sh --status` first.

**The UI loads but every level shows a red dot** — the page reached the server, the server did not reach the stack. `curl -s localhost:8800/api/health` names which one and why; it runs the same probes as `arc-rector doctor`. In the container the service URLs are compose names, not `localhost`, and every one of them is an env override you can see in `docker-compose.a1.yml`.

**The UI answers but the "Langfuse trace" link 404s** — `ARC_UI_TRACE_BASE` is unset or wrong. See above.

**A question times out on the UI** — a CPU-only ARM box takes minutes for a paragraph, and `ARC_L0_INFERENCE__TIMEOUT` (600s in the A1 profile) is the ceiling. The page shows the real error rather than a spinner that never stops. Lower `ARC_PIPELINE__TOP_K` to shorten the prompt, or cap the answer with `ARC_L0_INFERENCE__NUM_PREDICT=256`.

---

## Before this is more than a demo

Free hosting does not change the security posture. A quick tunnel is a public, unauthenticated endpoint in front of a stack with committed default credentials. Read **[PRODUCTION.md](PRODUCTION.md)** before pointing anything real at it — at minimum: regenerate every credential, put Cloudflare Access in front, and set rate limits.
