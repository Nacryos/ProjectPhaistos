"""Ugaritic experiment driver (Table 3 baseline integration)."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

from repro.paths import ARTIFACTS
from repro.reference.paper_metrics import TABLE3
from repro.utils import utc_now_iso, write_json
from repro.wrappers.neurodecipher_wrapper import run_neurodecipher


def run_reference() -> Dict:
    rows = [row for row in TABLE3 if row["lost"] == "Ugaritic"]
    return {
        "status": "ok",
        "mode": "reference",
        "created_at": utc_now_iso(),
        "rows": rows,
        "note": "Values are from the paper's reported table.",
    }


def _resolve_overrides(quick: bool) -> Optional[Dict[str, str]]:
    if quick:
        return {"num_rounds": "2", "num_epochs_per_M_step": "10"}
    overrides: Dict[str, str] = {}
    num_rounds = os.environ.get("UGA_NUM_ROUNDS")
    if num_rounds:
        overrides["num_rounds"] = num_rounds
    num_epochs = os.environ.get("UGA_NUM_EPOCHS_PER_M_STEP")
    if num_epochs:
        overrides["num_epochs_per_M_step"] = num_epochs
    return overrides or None


def run_neuro(quick: bool = False) -> Dict:
    overrides = _resolve_overrides(quick=quick)
    run = run_neurodecipher(cfg="UgaHebSmallNoSpe", overrides=overrides)
    rows = [row for row in TABLE3 if row["lost"] == "Ugaritic"]
    neuro_row = next((row for row in rows if row["method"] == "NeuroCipher"), None)
    return {
        "status": run.get("status", "failed"),
        "mode": "neuro",
        "created_at": utc_now_iso(),
        "rows": rows,
        "neuro_run": run,
        "paper_neuro_score": neuro_row["score"] if neuro_row else None,
        "note": "NeuroDecipher output can differ due to optimization and dependency/runtime drift.",
        "overrides": overrides or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ugaritic baseline experiment.")
    parser.add_argument("--mode", choices=["reference", "neuro"], default="reference")
    parser.add_argument("--quick", action="store_true", help="Use a short debug run (2 rounds x 10 epochs per M-step).")
    args = parser.parse_args()

    if args.mode == "reference":
        payload = run_reference()
    else:
        payload = run_neuro(quick=args.quick)

    out_path = ARTIFACTS / "runs" / "ugaritic_table3.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
