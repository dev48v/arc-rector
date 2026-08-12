#!/usr/bin/env bash
#
# Publish the Arc Rector demo through a Cloudflare tunnel.
#
#   ./deploy/tunnel.sh                 quick tunnel to the web UI (ephemeral URL)
#   ./deploy/tunnel.sh 3000            quick tunnel to Langfuse
#   ./deploy/tunnel.sh 6333            quick tunnel to the Qdrant dashboard
#   ./deploy/tunnel.sh 8800 named my-tunnel arc.example.com
#
#   ./deploy/tunnel.sh install-service install a systemd unit and enable it
#   ./deploy/tunnel.sh url             print the current public URL
#   ./deploy/tunnel.sh 8800 run        foreground; what the systemd unit calls
#
# The default port follows $ARC_PORT_UI, the same variable docker-compose.a1.yml
# publishes the UI on, so a box that had to move its ports does not have to
# remember a second number here.
#
# A tunnel started with `nohup` dies with your SSH session's process group and
# does not come back after a reboot, so anything you intend to leave up wants
# `install-service`. That unit runs `run` mode, which picks the named tunnel if
# its credentials exist and falls back to a quick one otherwise -- so creating
# the named tunnel later needs no change to the unit.
#
# Why a tunnel at all: the target VM's ingress is hardened to TCP 22 only. No
# security-list change, no opened port, no inbound firewall rule -- cloudflared
# makes an OUTBOUND connection and Cloudflare proxies traffic back down it.
#
# THIS BOX ALREADY RUNS A cloudflared FOR SOMETHING ELSE. This script starts a
# SEPARATE process with its own config and its own log. It must never restart,
# reconfigure, or stop the existing `wa-presence-cloudflared-1` container.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# `tunnel.sh url` and `tunnel.sh install-service` read better than
# `tunnel.sh 8800 url`, so a first argument that is not a port is a mode.
if [[ "${1:-}" =~ ^[^0-9] ]]; then
  set -- "${ARC_PORT_UI:-8800}" "$@"
fi

# The UI is the front door: it is the only one of these a person is meant to
# look at. Langfuse and Qdrant are still one argument away.
readonly PORT="${1:-${ARC_PORT_UI:-8800}}"
readonly MODE="${2:-quick}"
readonly TUNNEL_NAME="${3:-arc-rector}"
readonly HOSTNAME_ARG="${4:-}"
readonly LOG="${ARC_TUNNEL_LOG:-/tmp/arc-rector-tunnel.log}"
readonly PIDFILE="/tmp/arc-rector-tunnel.pid"
# Written every time a quick tunnel comes up, because the URL is different every
# time. `tunnel.sh url` reads this, and so does `a1-setup.sh --status`.
readonly URL_FILE="${ARC_TUNNEL_URL_FILE:-${REPO_DIR}/.arc_rector/tunnel-url.txt}"
readonly SERVICE_NAME="arc-rector-tunnel.service"
readonly UNIT_TEMPLATE="${REPO_DIR}/deploy/${SERVICE_NAME}"
readonly NAMED_CONFIG="${HOME}/.cloudflared/arc-rector-config.yml"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
info() { printf '  %s\n' "$*"; }
ok()   { printf '  %sok%s   %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '  %swarn%s %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '  %sabort%s %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- reporting
# `url` answers the only question anyone asks of a quick tunnel, and must work
# with the stack down, so it runs before every check below.
if [[ "$MODE" == "url" ]]; then
  [[ -s "$URL_FILE" ]] || die "no URL recorded yet at $URL_FILE. Is the tunnel running?"
  cat "$URL_FILE"
  exit 0
fi

printf '%s\n' "${BOLD}Arc Rector -- Cloudflare tunnel${NC}"

command -v cloudflared >/dev/null 2>&1 || die \
  "cloudflared is not installed. See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"

# ------------------------------------------------------------ as a service
# nohup does not survive a reboot and does not restart a crashed tunnel. This
# does both, and it is the ONLY thing here that needs root -- it writes one unit
# file under our own name and touches nothing else on the box.
if [[ "$MODE" == "install-service" ]]; then
  [[ -f "$UNIT_TEMPLATE" ]] || die "missing $UNIT_TEMPLATE"
  command -v systemctl >/dev/null 2>&1 || die "systemd is not available on this host."

  info "installing ${SERVICE_NAME} (user $(id -un), repo ${REPO_DIR}, port ${PORT})"
  sed -e "s#@USER@#$(id -un)#g" -e "s#@REPO_DIR@#${REPO_DIR}#g" -e "s#@PORT@#${PORT}#g" \
    "$UNIT_TEMPLATE" | sudo tee "/etc/systemd/system/${SERVICE_NAME}" >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$SERVICE_NAME"
  ok "installed and enabled; it will come back after a reboot"

  info "waiting for the public URL..."
  for _ in $(seq 1 30); do
    [[ -s "$URL_FILE" ]] && break
    sleep 2
  done
  if [[ -s "$URL_FILE" ]]; then
    printf '\n  %sPublic URL:%s  %s\n\n' "$BOLD" "$NC" "$(cat "$URL_FILE")"
  else
    warn "no URL yet. Check: journalctl -u ${SERVICE_NAME} -n 50"
  fi
  info "status: systemctl status ${SERVICE_NAME}"
  info "logs:   journalctl -u ${SERVICE_NAME} -f"
  info "stop:   sudo systemctl disable --now ${SERVICE_NAME}"
  exit 0
fi

# Do not disturb anyone else's tunnel.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'cloudflared'; then
  warn "another cloudflared is running in a container:"
  docker ps --format '    {{.Names}}\t{{.Status}}' | grep cloudflared
  warn "this script starts a SEPARATE process and leaves that one alone."
fi

# Refuse to publish a port with nothing behind it. 401 counts as answering: with
# ARC_UI_BASIC_AUTH_USER set the UI challenges everything, and a tunnel in front
# of a password prompt is exactly what we are trying to start.
probe() { curl -s -o /dev/null -m 5 -w '%{http_code}' "$1" 2>/dev/null || true; }
case "$(probe "http://127.0.0.1:${PORT}")" in
  200|401|403) ;;
  *) case "$(probe "http://127.0.0.1:${PORT}/api/public/health")" in
       200|401) ;;
       *) die "nothing is answering on 127.0.0.1:${PORT}. Start the stack first: ./deploy/a1-setup.sh" ;;
     esac ;;
esac
ok "local service on port ${PORT} is responding"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  die "our tunnel is already running (pid $(cat "$PIDFILE")). Stop it with: kill \$(cat $PIDFILE)"
fi

mkdir -p "$(dirname "$URL_FILE")"

case "$MODE" in
  # ------------------------------------------------------------------ run
  # Foreground, no pidfile, no nohup: systemd owns the process, restarts it if
  # it dies, and starts it again after a reboot. Named if its credentials are
  # there, quick if they are not -- so creating the named tunnel later is a
  # `systemctl restart`, not an edit.
  run)
    if [[ -f "$NAMED_CONFIG" ]]; then
      info "named tunnel config found at ${NAMED_CONFIG}"
      exec cloudflared tunnel --no-autoupdate --config "$NAMED_CONFIG" run
    fi

    warn "no named tunnel configured; running a QUICK tunnel (the URL changes on every restart)"
    : > "$URL_FILE"
    # The URL is only ever printed, once, in cloudflared's own output. Tee it
    # into a file on the way past so something other than a log can answer
    # "where is it right now". `2>&1 |` keeps this shell as the parent, which
    # is what systemd's Type=simple is watching.
    cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" 2>&1 |
      while IFS= read -r line; do
        printf '%s\n' "$line"
        if [[ ! -s "$URL_FILE" && "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
          printf '%s\n' "${BASH_REMATCH[1]}" > "$URL_FILE"
        fi
      done
    ;;

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

    printf '%s\n' "$url" > "$URL_FILE"
    printf '\n  %sPublic URL:%s  %s\n\n' "$BOLD" "$NC" "$url"
    info "log:  $LOG"
    info "url:  $URL_FILE"
    info "stop: kill \$(cat $PIDFILE)"
    warn "this URL dies with this process and is different next time."
    warn "for something you can put in writing: ./deploy/tunnel.sh install-service"
    if [[ -z "${ARC_UI_BASIC_AUTH_USER:-}" ]]; then
      warn "ARC_UI_BASIC_AUTH_USER is not set: this endpoint is UNAUTHENTICATED."
    fi
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
    printf 'https://%s\n' "$HOSTNAME_ARG" > "$URL_FILE"
    printf '\n  %sPublic URL:%s  https://%s\n\n' "$BOLD" "$NC" "$HOSTNAME_ARG"
    info "log:  $LOG"
    info "stop: kill \$(cat $PIDFILE)"
    info "To keep it up across reboots: ./deploy/tunnel.sh install-service"
    ;;

  *) die "unknown mode '$MODE'. Use quick, named, run, install-service or url." ;;
esac
