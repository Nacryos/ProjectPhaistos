"""Iberian experiment driver for Figure 4 data."""

from __future__ import annotations

import argparse

from repro.paths import ARTIFACTS
from repro.reference.paper_metrics import FIG4A_PAPER_TRACE, FIG4_CLOSENESS_TRACE
from repro.utils import utc_now_iso, write_json


def run_reference() -> dict:
    return {
        "status": "ok",
        "mode": "reference",
        "created_at": utc_now_iso(),
        "fig4a": FIG4A_PAPER_TRACE,
        "fig4_closeness": FIG4_CLOSENESS_TRACE,
        "note": "Figure points are digitized traces from Figure 4 in the paper.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Figure 4 data.")
    parser.add_argument("--mode", choices=["reference", "train"], default="reference")
    args = parser.parse_args()

    if args.mode == "train":
        raise SystemExit(
            "Iberian train mode is not yet wired because personal-name correspondences are in a PDF source. "
            "Use reference mode or provide machine-readable gold labels in data_external/."
        )

    payload = run_reference()
    out_path = ARTIFACTS / "runs" / "iberian_fig4.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
