#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-4}"
MAX_ITEMS="${MAX_ITEMS:-0}"
SEED="${SEED:-1234}"
WORKERS="${WORKERS:-8}"
RESUME="${RESUME:-1}"
UNDIRECTED="${UNDIRECTED:-0}"
CORPORA="${CORPORA:-all}"

ARGS=(
  --steps "$STEPS"
  --max-items "$MAX_ITEMS"
  --seed "$SEED"
  --workers "$WORKERS"
  --corpora "$CORPORA"
)

if [ "$RESUME" = "1" ]; then
  ARGS+=(--resume)
fi

if [ "$UNDIRECTED" = "1" ]; then
  ARGS+=(--undirected)
fi

"$PYTHON_BIN" -m repro.experiments.validation_all_pairs "${ARGS[@]}"
