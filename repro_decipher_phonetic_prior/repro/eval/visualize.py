"""Results visualization for PhoneticPriorModel experiments.

Produces matplotlib figures from experiment outputs.  Can be used as an
importable library or from the CLI:

    python -m repro.eval.visualize training-loss outputs/gothic/full/restarts/restart_01/metrics.json
    python -m repro.eval.visualize p-at-k outputs/iberian_names/p_at_k.csv
    python -m repro.eval.visualize branch-comparison outputs/validation_*/results.csv
    python -m repro.eval.visualize char-heatmap outputs/gothic/full/restarts/restart_01/char_distr.npz
    python -m repro.eval.visualize confusion outputs/gothic/full/restarts/restart_01/char_distr.npz
    python -m repro.eval.visualize all outputs/gothic
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Use a Unicode-capable font for IPA characters.
plt.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _default_output(input_path: Path, suffix: str) -> Path:
    return input_path.parent / f"{input_path.stem}_{suffix}.png"


# ---------------------------------------------------------------------------
# 1. Character distribution heatmap
# ---------------------------------------------------------------------------

def save_char_distr(model: Any, out_dir: Path) -> Path:
    """Save the char_distr matrix and character inventories to npz.

    Call after training to persist the learned distribution for
    post-hoc visualization.
    """
    import torch
    char_distr = model.compute_char_distr().detach().cpu().numpy()
    out_path = out_dir / "char_distr.npz"
    np.savez_compressed(
        out_path,
        char_distr=char_distr,
        lost_chars=np.array(model.lost_chars, dtype=object),
        known_chars=np.array(model.known_chars, dtype=object),
    )
    return out_path


def load_char_distr(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load char_distr.npz saved by ``save_char_distr``."""
    data = np.load(path, allow_pickle=True)
    return (
        data["char_distr"],
        list(data["lost_chars"]),
        list(data["known_chars"]),
    )


def plot_char_heatmap(
    char_distr: np.ndarray,
    lost_chars: List[str],
    known_chars: List[str],
    out_path: Path,
    *,
    title: str = "Character Distribution P(known|lost)",
    cmap: str = "YlOrRd",
    max_chars: int = 60,
) -> Path:
    """Plot P(known|lost) as an annotated heatmap."""
    K, L = char_distr.shape

    # Truncate if too large — keep chars with highest max probability.
    if L > max_chars:
        col_max = char_distr.max(axis=0)
        top_cols = np.argsort(col_max)[-max_chars:]
        top_cols.sort()
        char_distr = char_distr[:, top_cols]
        lost_chars = [lost_chars[i] for i in top_cols]
        L = max_chars
    if K > max_chars:
        row_max = char_distr.max(axis=1)
        top_rows = np.argsort(row_max)[-max_chars:]
        top_rows.sort()
        char_distr = char_distr[top_rows, :]
        known_chars = [known_chars[i] for i in top_rows]
        K = max_chars

    fig_w = max(6, 0.4 * L + 1.5)
    fig_h = max(4, 0.35 * K + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(char_distr, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(L))
    ax.set_xticklabels(lost_chars, fontsize=7, rotation=90)
    ax.set_yticks(range(K))
    ax.set_yticklabels(known_chars, fontsize=7)
    ax.set_xlabel("Lost characters")
    ax.set_ylabel("Known characters")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. P@K curve
# ---------------------------------------------------------------------------

def plot_p_at_k(
    p_at_k_csv: Path,
    out_path: Path,
    *,
    title: str = "P@K Comparison",
    value_key: str = "p_at_k_best",
) -> Path:
    """Plot P@K curves from p_at_k.csv."""
    rows = _read_csv(p_at_k_csv)
    variants = sorted(set(r["variant"] for r in rows))

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant in variants:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: int(r["k"]),
        )
        ks = [int(r["k"]) for r in vrows]
        vals = [float(r.get(value_key, r.get("p_at_k_best", 0))) for r in vrows]
        ax.plot(ks, vals, marker="o", label=variant)

    ax.set_xlabel("K")
    ax.set_ylabel("P@K")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Training loss curves
# ---------------------------------------------------------------------------

def plot_training_loss(
    metrics_json_path: Path,
    out_path: Path,
    *,
    title: str = "Training Curves",
    components: Sequence[str] = ("objective", "quality", "omega_cov", "omega_loss"),
) -> Path:
    """Plot training curves from a restart's metrics.json."""
    data = _read_json(metrics_json_path)
    history = data.get("history", [])
    if not history:
        raise ValueError(f"No training history in {metrics_json_path}")

    steps = [float(h["step"]) for h in history]
    n = len(components)
    cols = min(2, n)
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 3.5 * rows_n), squeeze=False)
    for idx, comp in enumerate(components):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        vals = [float(h.get(comp, 0)) for h in history]
        ax.plot(steps, vals, linewidth=1.2)
        ax.set_title(comp, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)

    # Hide unused subplots.
    for idx in range(n, rows_n * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_training_loss_multi(
    metrics_json_paths: Dict[str, Path],
    out_path: Path,
    *,
    component: str = "objective",
    title: str = "Training Objective Comparison",
) -> Path:
    """Overlay a single training component from multiple restarts/variants."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in sorted(metrics_json_paths.items()):
        data = _read_json(path)
        history = data.get("history", [])
        if not history:
            continue
        steps = [float(h["step"]) for h in history]
        vals = [float(h.get(component, 0)) for h in history]
        ax.plot(steps, vals, label=label, linewidth=1.0)

    ax.set_xlabel("step")
    ax.set_ylabel(component)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4. Cross-branch comparison bar chart
# ---------------------------------------------------------------------------

def plot_branch_comparison(
    results_csv_paths: Sequence[Path],
    out_path: Path,
    *,
    metric: str = "score_best",
    title: str = "P@10 Across Validation Branches",
) -> Path:
    """Grouped bar chart comparing scores across branches/variants."""
    all_rows: List[Dict[str, str]] = []
    for p in results_csv_paths:
        all_rows.extend(_read_csv(p))
    if not all_rows:
        raise ValueError("No data found in results CSVs.")

    branches = sorted(set(r.get("branch", r.get("known_language", "?")) for r in all_rows))
    variants = sorted(set(r["variant"] for r in all_rows))

    x = np.arange(len(branches))
    width = 0.8 / max(1, len(variants))

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(branches) * len(variants)), 5))
    for i, variant in enumerate(variants):
        vals = []
        errs = []
        for branch in branches:
            match = [r for r in all_rows if r.get("branch", r.get("known_language", "")) == branch and r["variant"] == variant]
            if match:
                vals.append(float(match[0].get(metric, 0)))
                errs.append(float(match[0].get("score_std", 0)))
            else:
                vals.append(0.0)
                errs.append(0.0)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=variant, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(branches, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Confusion-style matrix (top-N predictions per lost char)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    char_distr: np.ndarray,
    lost_chars: List[str],
    known_chars: List[str],
    out_path: Path,
    *,
    top_n: int = 3,
    title: str = "Top-3 Predicted Known Characters per Lost Character",
    max_lost_chars: int = 40,
) -> Path:
    """Table showing top-N known-character predictions per lost character."""
    K, L = char_distr.shape
    if L > max_lost_chars:
        col_max = char_distr.max(axis=0)
        top_cols = np.argsort(col_max)[-max_lost_chars:]
        top_cols.sort()
        char_distr = char_distr[:, top_cols]
        lost_chars = [lost_chars[i] for i in top_cols]
        L = max_lost_chars

    fig_h = max(4, 0.35 * L + 1)
    fig, ax = plt.subplots(figsize=(3 + 2 * top_n, fig_h))
    ax.set_axis_off()

    col_labels = ["Lost"] + [f"Pred {i+1}" for i in range(top_n)] + [f"P({i+1})" for i in range(top_n)]
    table_data = []
    for j in range(L):
        col = char_distr[:, j]
        top_idx = np.argsort(col)[-top_n:][::-1]
        row = [lost_chars[j]]
        row.extend(known_chars[i] for i in top_idx)
        row.extend(f"{col[i]:.3f}" for i in top_idx)
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(title, fontsize=10, pad=20)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. Auto-detect and generate all plots for an experiment
# ---------------------------------------------------------------------------

def plot_all_for_experiment(
    output_dir: Path,
    *,
    figures_dir: Optional[Path] = None,
) -> List[Path]:
    """Auto-detect experiment type and generate all applicable plots."""
    if figures_dir is None:
        figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # Training loss from any metrics.json with history
    for mj in sorted(output_dir.rglob("metrics.json")):
        data = _read_json(mj)
        if "history" in data and data["history"]:
            rel = mj.relative_to(output_dir).with_suffix("").as_posix().replace("/", "_")
            out = figures_dir / f"training_{rel}.png"
            try:
                plot_training_loss(mj, out)
                generated.append(out)
            except Exception:
                pass

    # P@K curves
    for pk in sorted(output_dir.rglob("p_at_k.csv")):
        out = figures_dir / "p_at_k.png"
        try:
            plot_p_at_k(pk, out)
            generated.append(out)
        except Exception:
            pass

    # Branch comparison
    results_csvs = sorted(output_dir.rglob("results.csv"))
    if results_csvs:
        out = figures_dir / "branch_comparison.png"
        try:
            plot_branch_comparison(results_csvs, out)
            generated.append(out)
        except Exception:
            pass

    # Char distribution heatmaps
    for npz in sorted(output_dir.rglob("char_distr.npz")):
        rel = npz.relative_to(output_dir).with_suffix("").as_posix().replace("/", "_")
        cd, lc, kc = load_char_distr(npz)
        out = figures_dir / f"heatmap_{rel}.png"
        try:
            plot_char_heatmap(cd, lc, kc, out)
            generated.append(out)
        except Exception:
            pass
        out2 = figures_dir / f"confusion_{rel}.png"
        try:
            plot_confusion_matrix(cd, lc, kc, out2)
            generated.append(out2)
        except Exception:
            pass

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate result visualizations for PhoneticPriorModel experiments."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tl = sub.add_parser("training-loss", help="Training loss curves.")
    p_tl.add_argument("input", help="Path to metrics.json with training history.")
    p_tl.add_argument("-o", "--output", default=None)

    p_pk = sub.add_parser("p-at-k", help="P@K curve plot.")
    p_pk.add_argument("input", help="Path to p_at_k.csv.")
    p_pk.add_argument("-o", "--output", default=None)

    p_bc = sub.add_parser("branch-comparison", help="Cross-branch bar chart.")
    p_bc.add_argument("input", nargs="+", help="Path(s) to results.csv.")
    p_bc.add_argument("-o", "--output", default=None)
    p_bc.add_argument("--metric", default="score_best")

    p_hm = sub.add_parser("char-heatmap", help="Character distribution heatmap.")
    p_hm.add_argument("input", help="Path to char_distr.npz.")
    p_hm.add_argument("-o", "--output", default=None)

    p_cf = sub.add_parser("confusion", help="Top-N predicted chars per lost char.")
    p_cf.add_argument("input", help="Path to char_distr.npz.")
    p_cf.add_argument("-o", "--output", default=None)
    p_cf.add_argument("--top-n", type=int, default=3)

    p_all = sub.add_parser("all", help="Auto-detect and generate all plots.")
    p_all.add_argument("input", help="Top-level experiment output directory.")
    p_all.add_argument("--figures-dir", default=None)

    args = parser.parse_args()

    if args.cmd == "training-loss":
        inp = Path(args.input)
        out = Path(args.output) if args.output else _default_output(inp, "training")
        plot_training_loss(inp, out)
        print(f"Saved: {out}")

    elif args.cmd == "p-at-k":
        inp = Path(args.input)
        out = Path(args.output) if args.output else _default_output(inp, "p_at_k")
        plot_p_at_k(inp, out)
        print(f"Saved: {out}")

    elif args.cmd == "branch-comparison":
        inputs = [Path(p) for p in args.input]
        out = Path(args.output) if args.output else _default_output(inputs[0], "branches")
        plot_branch_comparison(inputs, out, metric=args.metric)
        print(f"Saved: {out}")

    elif args.cmd == "char-heatmap":
        inp = Path(args.input)
        cd, lc, kc = load_char_distr(inp)
        out = Path(args.output) if args.output else _default_output(inp, "heatmap")
        plot_char_heatmap(cd, lc, kc, out)
        print(f"Saved: {out}")

    elif args.cmd == "confusion":
        inp = Path(args.input)
        cd, lc, kc = load_char_distr(inp)
        out = Path(args.output) if args.output else _default_output(inp, "confusion")
        plot_confusion_matrix(cd, lc, kc, out, top_n=args.top_n)
        print(f"Saved: {out}")

    elif args.cmd == "all":
        inp = Path(args.input)
        fig_dir = Path(args.figures_dir) if args.figures_dir else None
        generated = plot_all_for_experiment(inp, figures_dir=fig_dir)
        print(f"Generated {len(generated)} figures:")
        for p in generated:
            print(f"  {p}")


if __name__ == "__main__":
    main()
