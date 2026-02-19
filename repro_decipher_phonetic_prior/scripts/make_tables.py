#!/usr/bin/env python3
"""Build paper-style tables from outputs/ artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.reference.paper_metrics import TABLE3 as PAPER_TABLE3


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


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def to_markdown(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def to_latex(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str, label: str) -> str:
    cols = "l" * len(headers)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(_fmt(x) for x in row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def build_table2(output_root: Path, table_dir: Path) -> bool:
    src = output_root / "gothic" / "table2.csv"
    rows = read_csv(src)
    if not rows:
        print(f"[warn] skipping table2: missing {src}")
        return False

    sorted_rows = sorted(rows, key=lambda r: (int(float(r["whitespace_ratio"])), str(r["known_language"])))
    out_rows: List[Dict[str, Any]] = []
    md_rows: List[List[Any]] = []
    for r in sorted_rows:
        item = {
            "WR": int(float(r["whitespace_ratio"])),
            "Known": r["known_language"],
            "base": float(r["base"]) if r.get("base") else None,
            "partial": float(r["partial"]) if r.get("partial") else None,
            "full": float(r["full"]) if r.get("full") else None,
        }
        out_rows.append(item)
        md_rows.append([item["WR"], item["Known"], item["base"], item["partial"], item["full"]])

    write_csv(table_dir / "table2_paper.csv", out_rows)
    md = to_markdown(["WR", "Known", "base", "partial", "full"], md_rows)
    tex = to_latex(["WR", "Known", "base", "partial", "full"], md_rows, "Gothic main results (paper style)", "tab:gothic_main")
    (table_dir / "table2_paper.md").write_text(md, encoding="utf8")
    (table_dir / "table2_paper.tex").write_text(tex, encoding="utf8")
    return True


def build_table3(output_root: Path, table_dir: Path) -> bool:
    rows: List[Dict[str, Any]] = []

    # Paper reference rows.
    for r in PAPER_TABLE3:
        rows.append(
            {
                "lost": r["lost"],
                "known": r["known"],
                "method": r["method"],
                "metric": r["metric"],
                "score": float(r["score"]),
                "source": "paper",
            }
        )

    # Reproduced Ugaritic rows.
    uga_rows = read_csv(output_root / "ugaritic" / "table3_ugaritic.csv")
    for r in uga_rows:
        method = str(r.get("method", ""))
        if method in {"Bayesian", "NeuroCipher"}:
            continue
        score = r.get("score_best")
        if score in (None, ""):
            continue
        rows.append(
            {
                "lost": r.get("lost", "Ugaritic"),
                "known": r.get("known", "Hebrew"),
                "method": method,
                "metric": r.get("metric", "P@1"),
                "score": float(score),
                "source": "reproduced",
            }
        )

    # Reproduced Gothic base rows from table2 averages over WR.
    gothic_rows = read_csv(output_root / "gothic" / "table2.csv")
    if gothic_rows:
        by_known: Dict[str, List[float]] = {}
        for r in gothic_rows:
            known = str(r.get("known_language", ""))
            if not known or not r.get("base"):
                continue
            by_known.setdefault(known, []).append(float(r["base"]))
        for known, vals in sorted(by_known.items()):
            rows.append(
                {
                    "lost": "Gothic",
                    "known": known,
                    "method": "base",
                    "metric": "P@10",
                    "score": sum(vals) / len(vals),
                    "source": "reproduced",
                }
            )

    write_csv(table_dir / "table3_paper.csv", rows)
    md_rows = [[r["lost"], r["known"], r["method"], r["metric"], r["score"], r["source"]] for r in rows]
    md = to_markdown(["Lost", "Known", "Method", "Metric", "Score", "Source"], md_rows)
    tex = to_latex(["Lost", "Known", "Method", "Metric", "Score", "Source"], md_rows, "Comparison table (paper style)", "tab:comparison")
    (table_dir / "table3_paper.md").write_text(md, encoding="utf8")
    (table_dir / "table3_paper.tex").write_text(tex, encoding="utf8")
    return bool(rows)


def build_table4(output_root: Path, table_dir: Path) -> bool:
    src = output_root / "gothic" / "table4.csv"
    rows = read_csv(src)
    if not rows:
        print(f"[warn] skipping table4: missing {src}")
        return False

    out_rows: List[Dict[str, Any]] = []
    md_rows: List[List[Any]] = []
    for r in rows:
        item = {
            "IPA": str(r.get("ipa", "")),
            "Omega_loss": str(r.get("omega_loss", "")),
            "base": float(r["base"]) if r.get("base") else None,
            "partial": float(r["partial"]) if r.get("partial") else None,
            "full": float(r["full"]) if r.get("full") else None,
        }
        out_rows.append(item)
        md_rows.append([item["IPA"], item["Omega_loss"], item["base"], item["partial"], item["full"]])

    write_csv(table_dir / "table4_paper.csv", out_rows)
    md = to_markdown(["IPA", "Omega_loss", "base", "partial", "full"], md_rows)
    tex = to_latex(["IPA", "Omega_loss", "base", "partial", "full"], md_rows, "Ablation on Gothic-ON (paper style)", "tab:ablation")
    (table_dir / "table4_paper.md").write_text(md, encoding="utf8")
    (table_dir / "table4_paper.tex").write_text(tex, encoding="utf8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style tables from outputs/.")
    parser.add_argument("--output-root", default="outputs", help="Outputs directory root")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    ok2 = build_table2(output_root, table_dir)
    ok3 = build_table3(output_root, table_dir)
    ok4 = build_table4(output_root, table_dir)

    print(f"Wrote table files in {table_dir}")
    for name, ok in [("table2_paper.md", ok2), ("table3_paper.md", ok3), ("table4_paper.md", ok4)]:
        path = table_dir / name
        if ok and path.exists():
            print(path.read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
