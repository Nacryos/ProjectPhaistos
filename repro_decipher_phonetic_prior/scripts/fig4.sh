#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${IBERIAN_MODE:-reference}"

"$PYTHON_BIN" -m repro.experiments.iberian --mode "$MODE"
