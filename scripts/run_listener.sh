#!/usr/bin/env bash
set -euo pipefail

HOST="${PYSCRIBE_HOST:-127.0.0.1}"
PORT="${PYSCRIBE_PORT:-7860}"
MAX_PORT_TRIES="${PYSCRIBE_MAX_PORT_TRIES:-50}"
QUEUE_SIZE="${PYSCRIBE_QUEUE_SIZE:-16}"
ALLOW_NONLOCAL_HOST="${PYSCRIBE_ALLOW_NONLOCAL_HOST:-0}"
# Auth is read from environment in main.py to avoid leaking secrets via process args.

NONLOCAL_ARGS=()
if [[ "$ALLOW_NONLOCAL_HOST" == "1" ]]; then
  NONLOCAL_ARGS+=(--allow-nonlocal-host)
fi

exec python main.py serve \
  --host "$HOST" \
  --port "$PORT" \
  --max-port-tries "$MAX_PORT_TRIES" \
  --queue-size "$QUEUE_SIZE" \
  "${NONLOCAL_ARGS[@]}" \
  "$@"
