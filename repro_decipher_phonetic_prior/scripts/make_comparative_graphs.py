#!/usr/bin/env python3
"""Create direct reproduced-vs-paper comparison graphs from outputs/tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf8", newline="") as fin:
        return list(csv.DictReader(fin))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-vs-reproduced comparison graphs.")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    root = Path(args.output_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    table3 = read_csv(root / "tables" / "table3_paper.csv")
    if not table3:
        raise SystemExit(f"Missing comparison table: {root / 'tables' / 'table3_paper.csv'}")

    by_key: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for row in table3:
        key = (
            str(row.get("lost", "")),
            str(row.get("known", "")),
            str(row.get("method", "")),
            str(row.get("metric", "")),
        )
        score = float(row.get("score", "nan"))
        src = str(row.get("source", "")).strip().lower()
        by_key.setdefault(key, {})
        by_key[key][src] = score

    overlap_rows: List[Dict[str, Any]] = []
    for key, val in sorted(by_key.items()):
        if "paper" not in val or "reproduced" not in val:
            continue
        paper = float(val["paper"])
        rep = float(val["reproduced"])
        overlap_rows.append(
            {
                "lost": key[0],
                "known": key[1],
                "method": key[2],
                "metric": key[3],
                "paper_score": paper,
                "reproduced_score": rep,
                "delta_reproduced_minus_paper": rep - paper,
            }
        )

    write_csv(root / "tables" / "table3_overlap_comparison.csv", overlap_rows)

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Graph 1: side-by-side bars for overlapping entries only.
    fig1 = fig_dir / "table3_overlap_bars.png"
    if overlap_rows:
        labels = [f"{r['lost']}-{r['known']}-{r['method']}" for r in overlap_rows]
        x = list(range(len(labels)))
        paper_vals = [float(r["paper_score"]) for r in overlap_rows]
        rep_vals = [float(r["reproduced_score"]) for r in overlap_rows]

        width = 0.35
        plt.figure(figsize=(max(7, len(labels) * 1.2), 4))
        plt.bar([i - width / 2 for i in x], paper_vals, width=width, label="paper")
        plt.bar([i + width / 2 for i in x], rep_vals, width=width, label="reproduced")
        plt.ylim(0.0, 1.0)
        plt.ylabel("score")
        plt.title("Direct comparison (paper vs reproduced)")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig1, dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(7, 2.5))
        plt.axis("off")
        plt.text(0.5, 0.5, "No overlapping rows\n(paper and reproduced)", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig1, dpi=180)
        plt.close()

    # Graph 2: reproduced-vs-paper scatter for overlapping entries.
    fig2 = fig_dir / "table3_overlap_scatter.png"
    if overlap_rows:
        px = [float(r["paper_score"]) for r in overlap_rows]
        ry = [float(r["reproduced_score"]) for r in overlap_rows]
        labels = [f"{r['lost']}-{r['known']}-{r['method']}" for r in overlap_rows]

        plt.figure(figsize=(5, 5))
        plt.scatter(px, ry)
        for x, y, label in zip(px, ry, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("paper score")
        plt.ylabel("reproduced score")
        plt.title("Reproduced vs paper score")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(fig2, dpi=180)
        plt.close()
    else:
        plt.figure(figsize=(7, 2.5))
        plt.axis("off")
        plt.text(0.5, 0.5, "No overlapping rows for scatter", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig2, dpi=180)
        plt.close()

    summary_lines = []
    summary_lines.append("# Direct Comparison Summary\n")
    summary_lines.append(f"- Input table: `{root / 'tables' / 'table3_paper.csv'}`")
    summary_lines.append(f"- Overlap rows (paper + reproduced): {len(overlap_rows)}")
    if overlap_rows:
        mae = sum(abs(float(r["delta_reproduced_minus_paper"])) for r in overlap_rows) / len(overlap_rows)
        summary_lines.append(f"- Mean absolute delta: {mae:.3f}")
        worst = sorted(overlap_rows, key=lambda r: abs(float(r["delta_reproduced_minus_paper"])), reverse=True)[0]
        summary_lines.append(
            "- Largest delta: "
            f"{worst['lost']}/{worst['known']}/{worst['method']} "
            f"({float(worst['delta_reproduced_minus_paper']):+.3f})"
        )
    (root / "tables" / "COMPARISON_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf8")

    print(f"Wrote {root / 'tables' / 'table3_overlap_comparison.csv'}")
    print(f"Wrote {fig1}")
    print(f"Wrote {fig2}")
    print(f"Wrote {root / 'tables' / 'COMPARISON_SUMMARY.md'}")


if __name__ == "__main__":
    main()
