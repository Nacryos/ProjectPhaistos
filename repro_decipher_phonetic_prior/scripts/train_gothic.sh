#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${GOTHIC_MODE:-reference}"

"$PYTHON_BIN" -m repro.experiments.gothic --task table2 --mode "$MODE"
"$PYTHON_BIN" -m repro.experiments.gothic --task table4 --mode "$MODE"
