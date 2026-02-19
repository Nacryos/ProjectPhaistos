#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "${SKIP_SETUP:-1}" = "0" ]; then
  scripts/setup.sh
fi

scripts/prepare_datasets.sh

if [ "${USE_LEGACY_REFERENCE_PIPELINE:-0}" = "1" ]; then
  scripts/smoke_test.sh
  scripts/train_gothic.sh
  scripts/eval_gothic.sh
  scripts/fig4.sh
  python3 -m repro.report
  echo "Legacy reference pipeline complete."
else
  scripts/reproduce_paper.sh
  echo "Paper-style reproduction pipeline complete."
fi
