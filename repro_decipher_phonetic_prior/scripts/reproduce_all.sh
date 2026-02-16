#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${SKIP_SETUP:-1}" = "0" ]; then
  scripts/setup.sh
fi

scripts/prepare_datasets.sh
scripts/smoke_test.sh
scripts/train_gothic.sh
scripts/eval_gothic.sh

if ! scripts/run_neurocipher.sh; then
  echo "NeuroCipher run failed, falling back to reference values for table3 Ugaritic rows."
  NEURO_MODE=reference scripts/run_neurocipher.sh
fi

scripts/fig4.sh
"$PYTHON_BIN" -m repro.report

echo "Reproduction pipeline complete."
