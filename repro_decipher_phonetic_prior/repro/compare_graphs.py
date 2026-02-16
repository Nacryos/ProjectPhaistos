"""Create side-by-side comparison graphs against paper-reported values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from repro.paths import ARTIFACTS, FIGURES
from repro.reference.paper_metrics import TABLE2, TABLE3, TABLE4


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf8", newline="") as fin:
        return list(csv.DictReader(fin))


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def _mode_label() -> str:
    parts: List[str] = []
    gothic = _read_json(ARTIFACTS / "runs" / "gothic_table2.json")
    ug = _read_json(ARTIFACTS / "runs" / "ugaritic_table3.json")
    ib = _read_json(ARTIFACTS / "runs" / "iberian_fig4.json")
    if gothic:
        parts.append(f"Gothic={gothic.get('mode', 'unknown')}")
    if ug:
        parts.append(f"Ugaritic={ug.get('mode', 'unknown')}")
    if ib:
        parts.append(f"Iberian={ib.get('mode', 'unknown')}")
    return " | ".join(parts)


def _plot_table2_compare(out_path: Path) -> None:
    produced = _read_csv(ARTIFACTS / "table2.csv")
    if not produced:
        produced = [{k: str(v) for k, v in row.items()} for row in TABLE2["gothic"]]

    original_map: Dict[Tuple[int, str], Dict[str, float]] = {}
    for row in TABLE2["gothic"]:
        key = (int(row["whitespace_ratio"]), str(row["known_language"]))
        original_map[key] = {
            "base": float(row["base"]),
            "partial": float(row["partial"]),
            "full": float(row["full"]),
        }

    labels: List[str] = []
    orig_base: List[float] = []
    rep_base: List[float] = []
    orig_partial: List[float] = []
    rep_partial: List[float] = []
    orig_full: List[float] = []
    rep_full: List[float] = []

    for row in produced:
        key = (int(float(row["whitespace_ratio"])), str(row["known_language"]))
        labels.append(f"{key[0]}-{key[1]}")
        orig = original_map[key]
        orig_base.append(orig["base"])
        orig_partial.append(orig["partial"])
        orig_full.append(orig["full"])
        rep_base.append(_to_float(row.get("base")))
        rep_partial.append(_to_float(row.get("partial")))
        rep_full.append(_to_float(row.get("full")))

    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    specs = [
        ("base", orig_base, rep_base),
        ("partial", orig_partial, rep_partial),
        ("full", orig_full, rep_full),
    ]
    for ax, (name, original, reproduced) in zip(axes, specs):
        ax.bar(x - width / 2, original, width, label="paper original", color="#7b8aa0")
        ax.bar(x + width / 2, reproduced, width, label="reproduced", color="#d66a4f")
        ax.set_ylabel(f"{name} P@10")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right")
    fig.suptitle("Table 2: Gothic Original vs Reproduced")
    fig.text(0.01, 0.01, _mode_label(), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_table3_compare(out_path: Path) -> None:
    produced = _read_csv(ARTIFACTS / "table3.csv")
    original_map: Dict[Tuple[str, str, str, str], float] = {}
    for row in TABLE3:
        key = (row["lost"], row["known"], row["method"], row["metric"])
        original_map[key] = float(row["score"])

    labels: List[str] = []
    original_scores: List[float] = []
    reproduced_scores: List[float] = []
    has_rep: List[bool] = []

    for row in produced:
        key = (row["lost"], row["known"], row["method"], row["metric"])
        labels.append(f"{row['lost'][:3]}-{row['known'][:3]}:{row['method']}")
        original_scores.append(original_map.get(key, _to_float(row.get("score"))))
        rep = _to_float(row.get("score_reproduced"))
        reproduced_scores.append(rep)
        has_rep.append(not np.isnan(rep))

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, original_scores, width, label="paper original", color="#7b8aa0")

    rep_arr = np.array(reproduced_scores, dtype=float)
    mask = ~np.isnan(rep_arr)
    ax.bar(x[mask] + width / 2, rep_arr[mask], width, label="reproduced", color="#d66a4f")

    for idx, ok in enumerate(has_rep):
        if not ok:
            ax.text(x[idx] + width / 2, 0.02, "NA", rotation=90, va="bottom", ha="center", fontsize=8, color="#666666")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Table 3: Original vs Reproduced")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, _mode_label(), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_table4_compare(out_path: Path) -> None:
    produced = _read_csv(ARTIFACTS / "table4.csv")
    if not produced:
        produced = [{k: str(v) for k, v in row.items()} for row in TABLE4]

    original_map: Dict[Tuple[bool, bool], Dict[str, float]] = {}
    for row in TABLE4:
        key = (bool(row["ipa"]), bool(row["omega_loss"]))
        original_map[key] = {
            "base": float(row["base"]),
            "partial": float(row["partial"]),
            "full": float(row["full"]),
        }

    labels: List[str] = []
    orig_base: List[float] = []
    rep_base: List[float] = []
    orig_partial: List[float] = []
    rep_partial: List[float] = []
    orig_full: List[float] = []
    rep_full: List[float] = []

    def _as_bool(text: str) -> bool:
        return str(text).strip().lower() in {"1", "true", "yes"}

    for row in produced:
        key = (_as_bool(row["ipa"]), _as_bool(row["omega_loss"]))
        labels.append(f"ipa={int(key[0])},omega={int(key[1])}")
        orig = original_map[key]
        orig_base.append(orig["base"])
        orig_partial.append(orig["partial"])
        orig_full.append(orig["full"])
        rep_base.append(_to_float(row.get("base")))
        rep_partial.append(_to_float(row.get("partial")))
        rep_full.append(_to_float(row.get("full")))

    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    specs = [
        ("base", orig_base, rep_base),
        ("partial", orig_partial, rep_partial),
        ("full", orig_full, rep_full),
    ]
    for ax, (name, original, reproduced) in zip(axes, specs):
        ax.bar(x - width / 2, original, width, label="paper original", color="#7b8aa0")
        ax.bar(x + width / 2, reproduced, width, label="reproduced", color="#d66a4f")
        ax.set_ylabel(f"{name} P@10")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=0)
    fig.suptitle("Table 4: Original vs Reproduced")
    fig.text(0.01, 0.01, _mode_label(), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison plots against original paper numbers.")
    _ = parser.parse_args()

    _plot_table2_compare(FIGURES / "compare_table2.png")
    _plot_table3_compare(FIGURES / "compare_table3.png")
    _plot_table4_compare(FIGURES / "compare_table4.png")

    print(f"Wrote {FIGURES / 'compare_table2.png'}")
    print(f"Wrote {FIGURES / 'compare_table3.png'}")
    print(f"Wrote {FIGURES / 'compare_table4.png'}")


if __name__ == "__main__":
    main()
