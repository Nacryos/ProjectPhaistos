"""Gothic experiment driver for Table 2/4 and related metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from repro.paths import ARTIFACTS
from repro.reference.paper_metrics import TABLE2, TABLE3, TABLE4
from repro.utils import utc_now_iso, write_json


def run_reference(task: str) -> Dict:
    if task == "table2":
        rows = TABLE2["gothic"]
    elif task == "table4":
        rows = TABLE4
    elif task == "table3_gothic":
        rows = [row for row in TABLE3 if row["lost"] == "Gothic"]
    else:
        raise ValueError(f"Unsupported Gothic task: {task}")

    return {
        "status": "ok",
        "mode": "reference",
        "task": task,
        "created_at": utc_now_iso(),
        "rows": rows,
        "note": "Values are from the paper's reported tables.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gothic experiment wrappers.")
    parser.add_argument("--task", choices=["table2", "table4", "table3_gothic"], required=True)
    parser.add_argument("--mode", choices=["reference", "train"], default="reference")
    args = parser.parse_args()

    if args.mode == "train":
        raise SystemExit(
            "Gothic train mode requires external known-vocabulary assets (PG/ON/OE descendant-tree extracts). "
            "Run scripts/fetch_data.sh and provide converted vocab files, then re-run with wrapper extensions."
        )

    payload = run_reference(args.task)
    out_path = ARTIFACTS / "runs" / f"gothic_{args.task}.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
