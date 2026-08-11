#!/usr/bin/env bash
#
# Publish the Arc Rector demo through a Cloudflare tunnel.
#
#   ./deploy/tunnel.sh                 quick tunnel to the web UI (ephemeral URL)
#   ./deploy/tunnel.sh 3000            quick tunnel to Langfuse
#   ./deploy/tunnel.sh 6333            quick tunnel to the Qdrant dashboard
#   ./deploy/tunnel.sh 8800 named my-tunnel arc.example.com
#
# Why a tunnel at all: the target VM's ingress is hardened to TCP 22 only. No
# security-list change, no opened port, no inbound firewall rule -- cloudflared
# makes an OUTBOUND connection and Cloudflare proxies traffic back down it.
#
# THIS BOX ALREADY RUNS A cloudflared FOR SOMETHING ELSE. This script starts a
# SEPARATE process with its own config and its own log. It must never restart,
# reconfigure, or stop the existing `wa-presence-cloudflared-1` container.

set -euo pipefail

# The UI is the front door: it is the only one of these a person is meant to
# look at. Langfuse and Qdrant are still one argument away.
readonly PORT="${1:-8800}"
readonly MODE="${2:-quick}"
readonly TUNNEL_NAME="${3:-arc-rector}"
readonly HOSTNAME_ARG="${4:-}"
readonly LOG="${ARC_TUNNEL_LOG:-/tmp/arc-rector-tunnel.log}"
readonly PIDFILE="/tmp/arc-rector-tunnel.pid"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
info() { printf '  %s\n' "$*"; }
ok()   { printf '  %sok%s   %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '  %swarn%s %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '  %sabort%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

printf '%s\n' "${BOLD}Arc Rector -- Cloudflare tunnel${NC}"

command -v cloudflared >/dev/null 2>&1 || die \
  "cloudflared is not installed. See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"

# Do not disturb anyone else's tunnel.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'cloudflared'; then
  warn "another cloudflared is running in a container:"
  docker ps --format '    {{.Names}}\t{{.Status}}' | grep cloudflared
  warn "this script starts a SEPARATE process and leaves that one alone."
fi

# Refuse to publish a port with nothing behind it.
curl -fsS -m 5 "http://127.0.0.1:${PORT}" >/dev/null 2>&1 \
  || curl -fsS -m 5 "http://127.0.0.1:${PORT}/api/public/health" >/dev/null 2>&1 \
  || die "nothing is answering on 127.0.0.1:${PORT}. Start the stack first: ./deploy/a1-setup.sh"
ok "local service on port ${PORT} is responding"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  die "our tunnel is already running (pid $(cat "$PIDFILE")). Stop it with: kill \$(cat $PIDFILE)"
fi

case "$MODE" in
  # ---------------------------------------------------------------- quick
  # Ephemeral trycloudflare.com URL. No Cloudflare account, no DNS, no config.
  # The URL changes on every restart and the tunnel dies with this process, so
  # it is for demos and never for anything durable. It is also UNAUTHENTICATED:
  # anyone with the link reaches the service. Do not point it at real data.
  quick)
    info "mode: quick tunnel (ephemeral, unauthenticated)"
    : > "$LOG"
    nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" \
      >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    ok "cloudflared started (pid $(cat "$PIDFILE"))"

    info "waiting for the public URL..."
    url=""
    for _ in $(seq 1 30); do
      url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
      [[ -n "$url" ]] && break
      sleep 2
    done

    if [[ -z "$url" ]]; then
      warn "no URL appeared within 60s. Last log lines:"
      tail -20 "$LOG" | sed 's/^/    /'
      exit 1
    fi

    printf '\n  %sPublic URL:%s  %s\n\n' "$BOLD" "$NC" "$url"
    info "log:  $LOG"
    info "stop: kill \$(cat $PIDFILE)"
    warn "this URL is public and unauthenticated. It dies when the process does."
    ;;

  # ---------------------------------------------------------------- named
  # Stable hostname on a domain you control in Cloudflare. Survives restarts,
  # and can be put behind Cloudflare Access for real authentication -- which is
  # what you want for anything beyond a demo.
  named)
    [[ -n "$HOSTNAME_ARG" ]] || die "named mode needs a hostname: ./deploy/tunnel.sh 8800 named <tunnel-name> <hostname>"
    info "mode: named tunnel  ${TUNNEL_NAME} -> ${HOSTNAME_ARG}"

    if [[ ! -f "${HOME}/.cloudflared/cert.pem" ]]; then
      cat <<EOF

  One-time setup, in this order:

    1. cloudflared tunnel login
         Opens a browser URL; authorise the domain you own.

    2. cloudflared tunnel create ${TUNNEL_NAME}
         Writes ~/.cloudflared/<UUID>.json -- treat that file as a credential.

    3. cloudflared tunnel route dns ${TUNNEL_NAME} ${HOSTNAME_ARG}
         Creates the CNAME.

    4. Re-run this script.

  Then put it behind Cloudflare Access (Zero Trust > Access > Applications)
  so the endpoint is authenticated. Arc Rector has no auth of its own.

EOF
      die "not logged in to Cloudflare yet."
    fi

    cloudflared tunnel list 2>/dev/null | grep -q "\b${TUNNEL_NAME}\b" \
      || die "tunnel '${TUNNEL_NAME}' does not exist. Create it: cloudflared tunnel create ${TUNNEL_NAME}"

    local_cfg="${HOME}/.cloudflared/arc-rector-config.yml"
    tunnel_id=$(cloudflared tunnel list --output json 2>/dev/null \
      | grep -B2 "\"name\":\"${TUNNEL_NAME}\"" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4 || true)
    [[ -n "$tunnel_id" ]] || die "could not resolve the tunnel id for ${TUNNEL_NAME}."

    # Our own config file, so the existing cloudflared's config is untouched.
    cat > "$local_cfg" <<EOF
tunnel: ${tunnel_id}
credentials-file: ${HOME}/.cloudflared/${tunnel_id}.json
ingress:
  - hostname: ${HOSTNAME_ARG}
    service: http://127.0.0.1:${PORT}
  - service: http_status:404
EOF
    ok "wrote ${local_cfg}"

    : > "$LOG"
    nohup cloudflared tunnel --no-autoupdate --config "$local_cfg" run "$TUNNEL_NAME" >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    ok "cloudflared started (pid $(cat "$PIDFILE"))"
    printf '\n  %sPublic URL:%s  https://%s\n\n' "$BOLD" "$NC" "$HOSTNAME_ARG"
    info "log:  $LOG"
    info "stop: kill \$(cat $PIDFILE)"
    info "To run it as a service instead: sudo cloudflared service install"
    ;;

  *) die "unknown mode '$MODE'. Use 'quick' or 'named'." ;;
esac
