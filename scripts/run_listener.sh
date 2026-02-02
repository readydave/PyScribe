#!/usr/bin/env bash
set -euo pipefail

HOST="${PYSCRIBE_HOST:-0.0.0.0}"
PORT="${PYSCRIBE_PORT:-7860}"
MAX_PORT_TRIES="${PYSCRIBE_MAX_PORT_TRIES:-50}"
QUEUE_SIZE="${PYSCRIBE_QUEUE_SIZE:-16}"
AUTH_USER="${PYSCRIBE_AUTH_USER:-}"
AUTH_PASS="${PYSCRIBE_AUTH_PASS:-}"

AUTH_ARGS=()
if [[ -n "$AUTH_USER" && -n "$AUTH_PASS" ]]; then
  AUTH_ARGS+=(--auth-user "$AUTH_USER" --auth-pass "$AUTH_PASS")
fi

exec python main.py serve \
  --host "$HOST" \
  --port "$PORT" \
  --max-port-tries "$MAX_PORT_TRIES" \
  --queue-size "$QUEUE_SIZE" \
  "${AUTH_ARGS[@]}" \
  "$@"
