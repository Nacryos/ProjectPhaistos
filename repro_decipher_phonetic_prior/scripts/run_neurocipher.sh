#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${NEURO_MODE:-neuro}"
QUICK="${UGA_QUICK:-0}"

if [ "$QUICK" = "1" ]; then
  "$PYTHON_BIN" -m repro.experiments.ugaritic --mode "$MODE" --quick
else
  "$PYTHON_BIN" -m repro.experiments.ugaritic --mode "$MODE"
fi
