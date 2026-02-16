#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt

# Core local dependencies.
"$PYTHON_BIN" -m pip install -e third_party/arglib
"$PYTHON_BIN" -m pip install -e third_party/dev_misc
"$PYTHON_BIN" -m pip install -e third_party/DecipherUnsegmented
"$PYTHON_BIN" -m pip install -e third_party/NeuroDecipher

if [ "${SKIP_XIB:-0}" != "1" ]; then
  "$PYTHON_BIN" -m pip install -e third_party/xib
else
  echo "SKIP_XIB=1 -> skipping xib build/install"
fi

echo "Setup complete."
