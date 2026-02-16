#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BRANCH="${BRANCH:-germanic}"
LOST="${LOST:-got}"
KNOWN="${KNOWN:-ang}"
STEPS="${STEPS:-30}"
MAX_ITEMS="${MAX_ITEMS:-0}"
SEED="${SEED:-1234}"

"$PYTHON_BIN" -m repro.experiments.validation \
  --branch "$BRANCH" \
  --lost "$LOST" \
  --known "$KNOWN" \
  --steps "$STEPS" \
  --max-items "$MAX_ITEMS" \
  --seed "$SEED"
