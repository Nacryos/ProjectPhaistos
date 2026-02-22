#!/usr/bin/env bash
# ============================================================
# Lambda Cloud training script for PhoneticPriorModel
#
# Usage (inside tmux on Lambda):
#   bash scripts/lambda_train.sh           # run all experiments
#   bash scripts/lambda_train.sh ugaritic  # run just ugaritic
#   bash scripts/lambda_train.sh gothic    # run just gothic
#   bash scripts/lambda_train.sh iberian   # run just iberian names
#   bash scripts/lambda_train.sh parallel  # run ugaritic + iberian in parallel
#
# Quick start:
#   ssh <lambda-host>
#   tmux new -s train
#   git clone https://github.com/Nacryos/ProjectPhaistos.git && cd ProjectPhaistos
#   bash repro_decipher_phonetic_prior/scripts/lambda_train.sh
#   # Ctrl-b d to detach — reconnect later with: tmux attach -t train
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="$REPO_ROOT/repro_decipher_phonetic_prior"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_DIR/outputs}"
RESTARTS="${RESTARTS:-3}"
VARIANTS="${VARIANTS:-base,full}"
MODE="${1:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] OK:${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL:${NC} $*"; exit 1; }

# ── 1. Install dependencies ─────────────────────────────────
install_deps() {
    log "Installing Python dependencies..."
    pip install -q panphon pyyaml matplotlib scipy numpy torch tqdm 2>&1 | tail -1
    ok "Dependencies installed"
}

# ── 2. Verify environment ───────────────────────────────────
verify_env() {
    cd "$WORK_DIR"
    log "Verifying environment..."

    python -c "
import sys, torch
print(f'  Python:  {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)' if torch.cuda.is_available() else '  CUDA:    not available (CPU mode)')
"

    # Quick import check
    python -c "
from datasets.registry import get_corpus, list_corpora
for name in ['ugaritic', 'gothic', 'iberian']:
    c = get_corpus(name)
    print(f'  {name}: {len(c.lost_text)} training texts')
print(f'  Corpora OK ({len(list_corpora())} total)')
" || fail "Import check failed. Are you in the right directory?"

    ok "Environment verified"
}

# ── 3. Experiment runners ───────────────────────────────────
run_ugaritic() {
    log "Starting Ugaritic experiment ($RESTARTS restarts, variants=$VARIANTS)..."
    local t0=$(date +%s)

    python -m repro.run_experiment ugaritic \
        --restarts "$RESTARTS" \
        --variants "$VARIANTS" \
        --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/ugaritic_train.log"

    local elapsed=$(( $(date +%s) - t0 ))
    ok "Ugaritic completed in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
    echo ""

    # Show results if available
    if [ -f "$OUTPUT_ROOT/ugaritic/table3_ugaritic.csv" ]; then
        log "Ugaritic Results (Table 3):"
        column -t -s',' "$OUTPUT_ROOT/ugaritic/table3_ugaritic.csv" 2>/dev/null || cat "$OUTPUT_ROOT/ugaritic/table3_ugaritic.csv"
        echo ""
    fi
}

run_iberian() {
    log "Starting Iberian names experiment ($RESTARTS restarts, variants=$VARIANTS)..."
    local t0=$(date +%s)

    python -m repro.run_experiment iberian-names \
        --restarts "$RESTARTS" \
        --variants "$VARIANTS" \
        --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/iberian_train.log"

    local elapsed=$(( $(date +%s) - t0 ))
    ok "Iberian names completed in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
    echo ""

    if [ -f "$OUTPUT_ROOT/iberian_names/p_at_k.csv" ]; then
        log "Iberian P@K Results:"
        column -t -s',' "$OUTPUT_ROOT/iberian_names/p_at_k.csv" 2>/dev/null || cat "$OUTPUT_ROOT/iberian_names/p_at_k.csv"
        echo ""
    fi
}

run_gothic() {
    log "Starting Gothic experiment ($RESTARTS restarts, variants=$VARIANTS)..."
    log "This is the longest experiment — expect several hours."
    local t0=$(date +%s)

    python -m repro.run_experiment gothic \
        --restarts "$RESTARTS" \
        --variants "$VARIANTS" \
        --output-root "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/gothic_train.log"

    local elapsed=$(( $(date +%s) - t0 ))
    ok "Gothic completed in $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m"
    echo ""

    if [ -f "$OUTPUT_ROOT/gothic/table2.csv" ]; then
        log "Gothic Results (Table 2):"
        column -t -s',' "$OUTPUT_ROOT/gothic/table2.csv" 2>/dev/null || cat "$OUTPUT_ROOT/gothic/table2.csv"
        echo ""
    fi
}

run_smoke() {
    log "Running smoke test..."
    python -m repro.run_experiment ugaritic --smoke --restarts 1 --output-root "$OUTPUT_ROOT/smoke"
    ok "Smoke test passed"
}

# ── 4. Parallel mode ────────────────────────────────────────
run_parallel() {
    log "Starting parallel training (ugaritic + iberian in background, then gothic)..."
    mkdir -p "$OUTPUT_ROOT"

    # Run ugaritic and iberian in parallel background jobs
    run_ugaritic &
    local pid_uga=$!
    run_iberian &
    local pid_ibe=$!

    log "Ugaritic PID=$pid_uga, Iberian PID=$pid_ibe — waiting..."
    wait $pid_uga && ok "Ugaritic done" || warn "Ugaritic failed"
    wait $pid_ibe && ok "Iberian done" || warn "Iberian failed"

    # Gothic is the longest — run sequentially after the others finish
    run_gothic
}

# ── 5. Main ─────────────────────────────────────────────────
main() {
    echo ""
    echo "=========================================="
    echo " PhoneticPriorModel Training (Lambda)"
    echo "=========================================="
    echo "  Mode:       $MODE"
    echo "  Restarts:   $RESTARTS"
    echo "  Variants:   $VARIANTS"
    echo "  Output:     $OUTPUT_ROOT"
    echo "  Started:    $(date)"
    echo "=========================================="
    echo ""

    cd "$WORK_DIR"
    install_deps
    verify_env

    mkdir -p "$OUTPUT_ROOT"

    case "$MODE" in
        ugaritic)  run_ugaritic ;;
        iberian)   run_iberian ;;
        gothic)    run_gothic ;;
        smoke)     run_smoke ;;
        parallel)  run_parallel ;;
        all)
            run_ugaritic
            run_iberian
            run_gothic
            ;;
        *)
            fail "Unknown mode: $MODE (use: ugaritic, iberian, gothic, parallel, smoke, all)"
            ;;
    esac

    echo ""
    echo "=========================================="
    ok "All requested experiments finished at $(date)"
    echo "  Results in: $OUTPUT_ROOT"
    echo "=========================================="

    # List output files
    log "Output files:"
    find "$OUTPUT_ROOT" -name "*.csv" -o -name "*.json" | head -30
}

main
