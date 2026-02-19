#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESTARTS="${RESTARTS:-5}"
SEED_BASE="${SEED_BASE:-1234}"
MAX_QUERIES="${MAX_QUERIES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SMOKE="${SMOKE:-0}"

ARGS=(
  --variants "base,partial,full"
  --restarts "$RESTARTS"
  --seed-base "$SEED_BASE"
  --max-queries "$MAX_QUERIES"
  --output-root "$OUTPUT_ROOT"
)

if [ "$SMOKE" = "1" ]; then
  ARGS+=(--smoke)
fi

"$PYTHON_BIN" -m repro.run_experiment gothic "${ARGS[@]}"
