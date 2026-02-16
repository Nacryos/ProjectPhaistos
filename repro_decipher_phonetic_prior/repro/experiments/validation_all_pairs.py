"""Exhaustive validation-pair runs with tables and visual comparison outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets.registry import list_corpora, list_validation_languages
from repro.experiments.validation import run_validation
from repro.paths import ARTIFACTS, FIGURES
from repro.utils import utc_now_iso


@dataclass(frozen=True)
class PairTask:
    corpus: str
    branch: str
    lost: str
    known: str
    steps: int
    max_items: int
    seed: int


def _pair_seed(base_seed: int, branch: str, lost: str, known: str) -> int:
    text = f"{base_seed}:{branch}:{lost}:{known}"
    h = int(hashlib.sha256(text.encode("utf8")).hexdigest()[:8], 16)
    return (base_seed + h) % 2147483647


def _run_path(branch: str, lost: str, known: str) -> Path:
    return ARTIFACTS / "runs" / f"validation_{branch}_{lost}_to_{known}.json"


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _row_from_result(task: PairTask, payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "corpus": task.corpus,
        "branch": task.branch,
        "lost": task.lost,
        "known": task.known,
        "status": payload.get("status", "unknown"),
        "variant": payload.get("variant"),
        "num_lost_tokens": payload.get("num_lost_tokens"),
        "num_known_vocab": payload.get("num_known_vocab"),
        "steps": payload.get("steps", task.steps),
        "seed": payload.get("seed", task.seed),
        "p_at_1": _safe_float(payload.get("p_at_1")),
        "mrr": _safe_float(payload.get("mrr")),
        "top1_false_positive_rate": _safe_float(payload.get("top1_false_positive_rate")),
        "auroc": _safe_float(payload.get("auroc")),
        "score_margin": _safe_float(payload.get("score_margin")),
        "mean_positive_score": _safe_float(payload.get("mean_positive_score")),
        "mean_negative_score": _safe_float(payload.get("mean_negative_score")),
        "n_eval": payload.get("n_eval"),
        "n_pos_scored": payload.get("n_pos_scored"),
        "n_neg_scored": payload.get("n_neg_scored"),
        "run_file": str(_run_path(task.branch, task.lost, task.known)),
        "created_at": payload.get("created_at"),
    }


def _load_result(task: PairTask) -> Optional[Dict[str, object]]:
    path = _run_path(task.branch, task.lost, task.known)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def _execute_task(task: PairTask) -> Dict[str, object]:
    payload = run_validation(
        branch=task.branch,
        lost=task.lost,
        known=task.known,
        variant=None,
        steps=task.steps,
        max_items=task.max_items,
        seed=task.seed,
    )
    return _row_from_result(task, payload)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Sequence[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _language_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    langs = sorted({str(r["lost"]) for r in rows} | {str(r["known"]) for r in rows})
    summary: List[Dict[str, object]] = []
    for lang in langs:
        as_lost = [r for r in rows if str(r["lost"]) == lang]
        as_known = [r for r in rows if str(r["known"]) == lang]
        summary.append(
            {
                "language": lang,
                "n_pairs_as_lost": len(as_lost),
                "n_pairs_as_known": len(as_known),
                "mean_p_at_1_as_lost": _mean([_safe_float(r["p_at_1"]) for r in as_lost]),
                "mean_mrr_as_lost": _mean([_safe_float(r["mrr"]) for r in as_lost]),
                "mean_fp_rate_as_lost": _mean([_safe_float(r["top1_false_positive_rate"]) for r in as_lost]),
                "mean_auroc_as_lost": _mean([_safe_float(r["auroc"]) for r in as_lost]),
                "mean_p_at_1_as_known": _mean([_safe_float(r["p_at_1"]) for r in as_known]),
            }
        )
    return summary


def _matrix(rows: List[Dict[str, object]], value_key: str) -> Tuple[List[str], np.ndarray]:
    langs = sorted({str(r["lost"]) for r in rows} | {str(r["known"]) for r in rows})
    idx = {lang: i for i, lang in enumerate(langs)}
    mat = np.full((len(langs), len(langs)), np.nan, dtype=np.float32)
    for row in rows:
        i = idx[str(row["lost"])]
        j = idx[str(row["known"])]
        mat[i, j] = _safe_float(row[value_key])
    return langs, mat


def _plot_heatmap(
    labels: List[str],
    mat: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(mat, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Known language")
    ax.set_ylabel("Lost language")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Score", rotation=90)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(rows: List[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    branches = sorted({str(r["branch"]) for r in rows})
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(branches))))
    cmap = {branch: colors[i] for i, branch in enumerate(branches)}
    for branch in branches:
        b_rows = [r for r in rows if str(r["branch"]) == branch]
        xs = [_safe_float(r["p_at_1"]) for r in b_rows]
        ys = [_safe_float(r["top1_false_positive_rate"]) for r in b_rows]
        ax.scatter(xs, ys, s=22, alpha=0.75, label=branch, color=cmap[branch])
    ax.set_xlabel("P@1 (higher is better)")
    ax.set_ylabel("Top-1 false positive rate (lower is better)")
    ax.set_title("Validation Pair Quality vs False Positives")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_markdown(rows: List[Dict[str, object]], language_rows: List[Dict[str, object]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# VALIDATION_ALL_PAIRS_SUMMARY")
    lines.append("")
    lines.append(f"Generated at: {utc_now_iso()}")
    lines.append(f"Total pair runs: {len(rows)}")
    lines.append("")
    done = [r for r in rows if str(r.get("status")) == "ok"]
    lines.append(f"Successful runs: {len(done)}")
    if done:
        best = sorted(done, key=lambda r: _safe_float(r["p_at_1"]), reverse=True)[:10]
        worst = sorted(done, key=lambda r: _safe_float(r["p_at_1"]))[:10]
        lines.append("")
        lines.append("## Top 10 P@1 Pairs")
        for row in best:
            lines.append(
                f"- {row['branch']} {row['lost']} -> {row['known']}: "
                f"P@1={_safe_float(row['p_at_1']):.3f}, "
                f"FPR={_safe_float(row['top1_false_positive_rate']):.3f}, "
                f"AUROC={_safe_float(row['auroc']):.3f}"
            )
        lines.append("")
        lines.append("## Bottom 10 P@1 Pairs")
        for row in worst:
            lines.append(
                f"- {row['branch']} {row['lost']} -> {row['known']}: "
                f"P@1={_safe_float(row['p_at_1']):.3f}, "
                f"FPR={_safe_float(row['top1_false_positive_rate']):.3f}, "
                f"AUROC={_safe_float(row['auroc']):.3f}"
            )

    lines.append("")
    lines.append("## Best Languages as Lost (mean P@1)")
    best_langs = sorted(language_rows, key=lambda r: _safe_float(r["mean_p_at_1_as_lost"]), reverse=True)[:10]
    for row in best_langs:
        lines.append(
            f"- {row['language']}: mean P@1={_safe_float(row['mean_p_at_1_as_lost']):.3f}, "
            f"mean FPR={_safe_float(row['mean_fp_rate_as_lost']):.3f}, "
            f"pairs={row['n_pairs_as_lost']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _build_tasks(
    corpora: Sequence[str],
    directed: bool,
    steps: int,
    max_items: int,
    seed: int,
) -> List[PairTask]:
    tasks: List[PairTask] = []
    for corpus in corpora:
        branch = corpus.replace("validation_", "", 1)
        langs = list_validation_languages(corpus)
        if directed:
            for lost in langs:
                for known in langs:
                    if known == lost:
                        continue
                    tasks.append(
                        PairTask(
                            corpus=corpus,
                            branch=branch,
                            lost=lost,
                            known=known,
                            steps=steps,
                            max_items=max_items,
                            seed=_pair_seed(seed, branch, lost, known),
                        )
                    )
        else:
            for i, lost in enumerate(langs):
                for known in langs[i + 1:]:
                    tasks.append(
                        PairTask(
                            corpus=corpus,
                            branch=branch,
                            lost=lost,
                            known=known,
                            steps=steps,
                            max_items=max_items,
                            seed=_pair_seed(seed, branch, lost, known),
                        )
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exhaustive validation-pair comparisons with plots.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true", help="Reuse existing run JSON files when available.")
    parser.add_argument("--undirected", action="store_true", help="Run each language pair only once per corpus.")
    parser.add_argument(
        "--corpora",
        default="all",
        help="Comma-separated validation corpora names or 'all'. Example: validation_germanic,validation_names",
    )
    args = parser.parse_args()

    available = [c for c in list_corpora() if c.startswith("validation_")]
    if args.corpora == "all":
        corpora = available
    else:
        corpora = [c.strip().lower() for c in args.corpora.split(",") if c.strip()]
        for corpus in corpora:
            if corpus not in available:
                raise ValueError(f"Unknown validation corpus {corpus!r}. Choices: {available}")

    tasks = _build_tasks(
        corpora=corpora,
        directed=not args.undirected,
        steps=args.steps,
        max_items=args.max_items,
        seed=args.seed,
    )

    rows: List[Dict[str, object]] = []
    pending: List[PairTask] = []
    for task in tasks:
        cached = _load_result(task) if args.resume else None
        if cached:
            rows.append(_row_from_result(task, cached))
        else:
            pending.append(task)

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_task = {ex.submit(_execute_task, task): task for task in pending}
            done_count = 0
            for fut in as_completed(fut_to_task):
                task = fut_to_task[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "corpus": task.corpus,
                        "branch": task.branch,
                        "lost": task.lost,
                        "known": task.known,
                        "status": f"failed:{exc}",
                        "steps": task.steps,
                        "seed": task.seed,
                        "run_file": str(_run_path(task.branch, task.lost, task.known)),
                    }
                rows.append(row)
                done_count += 1
                if done_count % 5 == 0 or done_count == len(pending):
                    print(f"Completed {done_count}/{len(pending)} new pair runs")

    rows.sort(key=lambda r: (str(r["corpus"]), str(r["lost"]), str(r["known"])))
    table_path = ARTIFACTS / "validation_all_pairs.csv"
    _write_csv(table_path, rows)

    language_rows = _language_summary(rows)
    language_path = ARTIFACTS / "validation_language_summary.csv"
    _write_csv(language_path, language_rows)

    names_rows = [r for r in rows if str(r["corpus"]) == "validation_names"]
    if names_rows:
        labels, mat = _matrix(names_rows, "p_at_1")
        _plot_heatmap(
            labels,
            mat,
            "Validation Names: Directed P@1 Matrix",
            FIGURES / "validation_names_p_at_1_heatmap.png",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        labels, mat = _matrix(names_rows, "top1_false_positive_rate")
        _plot_heatmap(
            labels,
            mat,
            "Validation Names: Directed Top-1 False Positive Rate",
            FIGURES / "validation_names_fp_rate_heatmap.png",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )

    _plot_scatter(rows, FIGURES / "validation_all_pairs_scatter.png")
    _write_markdown(rows, language_rows, ARTIFACTS / "VALIDATION_ALL_PAIRS_SUMMARY.md")

    print(f"Wrote {table_path}")
    print(f"Wrote {language_path}")
    if names_rows:
        print(f"Wrote {FIGURES / 'validation_names_p_at_1_heatmap.png'}")
        print(f"Wrote {FIGURES / 'validation_names_fp_rate_heatmap.png'}")
    print(f"Wrote {FIGURES / 'validation_all_pairs_scatter.png'}")
    print(f"Wrote {ARTIFACTS / 'VALIDATION_ALL_PAIRS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
