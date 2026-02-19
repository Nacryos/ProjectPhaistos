#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

scripts/train_gothic.sh
python3 scripts/make_tables.py --output-root "${OUTPUT_ROOT:-outputs}"
