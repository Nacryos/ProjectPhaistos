#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESTARTS="${RESTARTS:-5}"
SEED_BASE="${SEED_BASE:-1234}"
MAX_QUERIES="${MAX_QUERIES:-0}"
SMOKE="${SMOKE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

COMMON_ARGS=(
  --restarts "$RESTARTS"
  --seed-base "$SEED_BASE"
  --max-queries "$MAX_QUERIES"
  --output-root "$OUTPUT_ROOT"
)

if [ "$SMOKE" = "1" ]; then
  COMMON_ARGS+=(--smoke)
fi

"$PYTHON_BIN" -m repro.run_experiment gothic --variants base,partial,full "${COMMON_ARGS[@]}"
"$PYTHON_BIN" -m repro.run_experiment ugaritic --variants base,full "${COMMON_ARGS[@]}"
"$PYTHON_BIN" -m repro.run_experiment iberian-names --variants base,full "${COMMON_ARGS[@]}"
"$PYTHON_BIN" -m repro.run_experiment iberian-closeness --variants base,full "${COMMON_ARGS[@]}"
"$PYTHON_BIN" scripts/make_tables.py --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" scripts/make_comparative_graphs.py --output-root "$OUTPUT_ROOT"

echo "Paper-style reproduction complete. Outputs at $ROOT_DIR/$OUTPUT_ROOT"
