"""Paper-style closeness/coverage-confidence experiments (Figure 4b/c/d style)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from datasets.registry import get_corpus
from repro.eval.common import (
    BilingualDataset,
    MissingDataError,
    TrainVariant,
    apply_whitespace_ratio,
    build_curve,
    check_sanity,
    closeness_score,
    compute_metrics,
    load_bilingual_dataset,
    load_plain_vocab,
    load_vocab_from_xml,
    load_yaml,
    now_meta,
    rank_queries,
    ranking_records_to_rows,
    resolve_path,
    restart_seeds,
    summarize_restarts,
    train_model,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_LOST_COLS = ["lost", "source", "query", "gothic", "uga-no_spe", "iberian", "lost_stem"]
DEFAULT_KNOWN_COLS = ["known", "target", "cognate", "known_stem", "hebrew", "latin"]


def _load_variants(cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[TrainVariant]:
    variants_cfg = cfg.get("variants", {})
    if not variants_cfg:
        raise ValueError("No variants defined in Iberian config.")
    names = list(selected) if selected else list(variants_cfg.keys())
    out: List[TrainVariant] = []
    for name in names:
        spec = variants_cfg.get(name)
        if spec is None:
            raise ValueError(f"Unknown variant {name!r}")
        out.append(
            TrainVariant(
                name=name,
                ipa=bool(spec.get("ipa", True)),
                omega_loss=bool(spec.get("omega_loss", True)),
                mapping_prior=str(spec.get("mapping_prior", "none")),
                partial_chars=[str(x).lower() for x in spec.get("partial_chars", [])],
                prior_strength=float(spec.get("prior_strength", 8.0)),
            )
        )
    return out


def _load_candidate_vocab(spec: Mapping[str, Any], max_items: int) -> Optional[List[str]]:
    source = str(spec.get("source", "plain_vocab"))
    required = bool(spec.get("required", True))
    path_value = spec.get("path")
    if not path_value:
        if required:
            raise MissingDataError("Candidate vocab entry missing path.")
        return None

    path = resolve_path(str(path_value))
    if not path.exists():
        if required:
            raise MissingDataError(f"Missing candidate vocabulary file: {path}")
        return None

    min_len = int(spec.get("min_len", 2))

    if source in {"plain_vocab", "txt"}:
        vocab = load_plain_vocab(path, min_len=min_len)
    elif source in {"xml_vocab", "xml"}:
        vocab = load_vocab_from_xml(path, min_len=min_len, max_items=max_items)
    elif source in {"bilingual_tsv", "cognate_tsv", "names_tsv"}:
        ds = load_bilingual_dataset(
            name="candidate_vocab",
            path=path,
            lost_col_candidates=[str(x) for x in spec.get("lost_col_candidates", DEFAULT_LOST_COLS)],
            known_col_candidates=[str(x) for x in spec.get("known_col_candidates", DEFAULT_KNOWN_COLS)],
            min_lost_len=int(spec.get("min_lost_len", 1)),
            min_known_len=min_len,
        )
        vocab = ds.known_vocab
    else:
        raise ValueError(f"Unsupported candidate vocabulary source: {source}")

    if max_items > 0:
        vocab = vocab[:max_items]
    return vocab


def _load_eval_dataset(spec: Mapping[str, Any], max_queries: int) -> Optional[BilingualDataset]:
    path_value = spec.get("path")
    if not path_value:
        return None
    path = resolve_path(str(path_value))
    if not path.exists():
        if bool(spec.get("required", False)):
            raise MissingDataError(f"Missing eval pair file: {path}")
        return None

    ds = load_bilingual_dataset(
        name="closeness_eval",
        path=path,
        lost_col_candidates=[str(x) for x in spec.get("lost_col_candidates", DEFAULT_LOST_COLS)],
        known_col_candidates=[str(x) for x in spec.get("known_col_candidates", DEFAULT_KNOWN_COLS)],
        min_lost_len=int(spec.get("min_lost_len", 1)),
        min_known_len=int(spec.get("min_known_len", 2)),
    )
    if max_queries > 0:
        kept = ds.lost_queries[:max_queries]
        ds = BilingualDataset(
            name=ds.name,
            lost_queries=kept,
            gold_map={q: ds.gold_map[q] for q in kept},
            known_vocab=ds.known_vocab,
            metadata=dict(ds.metadata),
        )
    return ds


def run_iberian_closeness(
    *,
    config_path: Path,
    output_root: Path,
    variants: Optional[Sequence[str]],
    restarts: int,
    seed_base: int,
    max_queries: int,
    smoke: bool,
    corpus_variant: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    train_cfg = cfg.get("training", {})
    if smoke:
        restarts = 1
        max_queries = 50 if max_queries <= 0 else min(max_queries, 50)
        train_cfg = dict(train_cfg)
        train_cfg["num_steps"] = int(cfg.get("smoke_num_steps", 20))
        smoke_train_lines_max = int(cfg.get("smoke_train_lines_max", 32))
    else:
        smoke_train_lines_max = 0
    smoke_known_vocab_max = int(cfg.get("smoke_known_vocab_max", 0)) if smoke else 0

    thresholds = [float(x) for x in cfg.get("closeness_thresholds", [i / 20.0 for i in range(21)])]
    variant_specs = _load_variants(cfg, variants)

    targets_cfg = cfg.get("closeness_targets", {})
    if not targets_cfg:
        raise ValueError("No closeness_targets defined in Iberian config.")

    out_dir = output_root / "iberian_closeness"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variant_specs:
        v_dir = out_dir / variant.name
        v_dir.mkdir(parents=True, exist_ok=True)

        curve_rows: List[Dict[str, Any]] = []
        ranking_rows: List[Dict[str, Any]] = []

        for target_name, target_spec in targets_cfg.items():
            lost_corpus_name = str(target_spec.get("lost_corpus", target_name))
            corpus = get_corpus(lost_corpus_name, variant=corpus_variant)
            train_text = corpus.lost_text
            if not train_text:
                raise MissingDataError(f"Lost corpus {lost_corpus_name!r} has empty text for closeness.")
            if smoke_train_lines_max > 0:
                train_text = train_text[:smoke_train_lines_max]

            wr = int(target_spec.get("whitespace_ratio", 100))
            candidate_max_items = int(target_spec.get("candidate_vocab_max_items", 0))
            eval_query_max = int(target_spec.get("eval_query_max", max_queries if max_queries > 0 else 0))

            candidate_cfg = target_spec.get("candidate_languages", {})
            eval_pairs_cfg = target_spec.get("eval_pairs", {})
            if not candidate_cfg:
                raise ValueError(f"No candidate_languages specified for closeness target {target_name!r}")

            for lang_label, lang_spec in candidate_cfg.items():
                known_vocab = _load_candidate_vocab(lang_spec, max_items=candidate_max_items)
                if known_vocab is None:
                    continue
                if smoke_known_vocab_max > 0 and len(known_vocab) > smoke_known_vocab_max:
                    known_vocab = known_vocab[:smoke_known_vocab_max]

                eval_ds = _load_eval_dataset(eval_pairs_cfg.get(lang_label, {}), max_queries=eval_query_max)
                if eval_ds is not None:
                    queries = eval_ds.lost_queries
                    gold_map = eval_ds.gold_map
                else:
                    # Fallback unsupervised query set from lost corpus lines.
                    queries = [x for x in train_text if x][: (eval_query_max or len(train_text))]
                    gold_map = {q: [] for q in queries}

                restart_rows: List[Dict[str, Any]] = []
                restart_curve_rows: List[List[Dict[str, Any]]] = []
                restart_query_rows: List[List[Dict[str, Any]]] = []

                for restart_idx, seed in enumerate(restart_seeds(seed_base, restarts), start=1):
                    run_dir = v_dir / "restarts" / f"{target_name}_{lang_label}_restart_{restart_idx:02d}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    train_text_ratio = apply_whitespace_ratio(train_text, wr, seed=seed)
                    if max_queries > 0:
                        train_text_ratio = train_text_ratio[: max_queries * 8]

                    train_out = train_model(
                        lost_training_text=train_text_ratio,
                        known_vocab=known_vocab,
                        variant=variant,
                        seed=seed,
                        train_cfg=train_cfg,
                    )
                    records = rank_queries(
                        model=train_out.model,
                        queries=queries,
                        gold_map=gold_map,
                        known_vocab=known_vocab,
                        top_k=10,
                    )
                    metrics = compute_metrics(records, ks=[1, 10])
                    curve = build_curve(records, thresholds)
                    check_sanity(records, curve_rows=curve)
                    confs = [float(r.top1_confidence) for r in records]
                    close = closeness_score(curve, confs)

                    rows = ranking_records_to_rows(records, top_k=10)
                    write_csv(run_dir / "per_query.csv", rows)
                    write_jsonl(run_dir / "per_query.jsonl", rows)
                    write_csv(run_dir / "curve.csv", curve)
                    write_json(
                        run_dir / "metrics.json",
                        {
                            **now_meta(),
                            "experiment": "iberian_closeness",
                            "variant": variant.name,
                            "target": target_name,
                            "candidate_language": lang_label,
                            "seed": seed,
                            "metrics": metrics,
                            "closeness": close,
                            "history": train_out.history,
                        },
                    )

                    restart_rows.append(
                        {
                            "seed": seed,
                            "p_at_1": float(metrics.get("p_at_1", 0.0)),
                            "mrr": float(metrics.get("mrr", 0.0)),
                            "mean_confidence": float(close["mean_confidence"]),
                            "auc_coverage": float(close["auc_coverage"]),
                            "auc_weighted_accuracy": float(close["auc_weighted_accuracy"]),
                            "closeness": float(close["closeness"]),
                        }
                    )
                    restart_curve_rows.append(curve)
                    restart_query_rows.append(rows)

                summary = summarize_restarts(restart_rows, primary_metric="closeness")
                best_idx = next((i for i, r in enumerate(restart_rows) if r["seed"] == summary["best_seed"]), 0)
                best_curve = restart_curve_rows[best_idx]
                best_queries = restart_query_rows[best_idx]

                lang_dir = v_dir / f"{target_name}_{lang_label}"
                lang_dir.mkdir(parents=True, exist_ok=True)
                write_csv(lang_dir / "restart_summary.csv", restart_rows)
                write_csv(lang_dir / "curve.csv", best_curve)
                write_csv(lang_dir / "per_query.csv", best_queries)
                write_jsonl(lang_dir / "per_query.jsonl", best_queries)
                write_json(
                    lang_dir / "metrics.json",
                    {
                        **now_meta(),
                        "experiment": "iberian_closeness",
                        "variant": variant.name,
                        "target": target_name,
                        "candidate_language": lang_label,
                        "summary": summary,
                        "best_restart": restart_rows[best_idx],
                        "restarts": restarts,
                    },
                )

                for row in best_curve:
                    curve_rows.append(
                        {
                            "variant": variant.name,
                            "target": target_name,
                            "candidate_language": lang_label,
                            "threshold": float(row["threshold"]),
                            "coverage": float(row["coverage"]),
                            "accuracy": float(row["accuracy"]) if not math.isnan(float(row["accuracy"])) else float("nan"),
                            "best_seed": int(summary["best_seed"]),
                        }
                    )

                ranking_rows.append(
                    {
                        "variant": variant.name,
                        "target": target_name,
                        "candidate_language": lang_label,
                        "closeness": float(summary["best"]),
                        "closeness_mean": float(summary["mean"]),
                        "closeness_std": float(summary["std"]),
                        "best_seed": int(summary["best_seed"]),
                        "p_at_1": float(restart_rows[best_idx]["p_at_1"]),
                        "mrr": float(restart_rows[best_idx]["mrr"]),
                        "mean_confidence": float(restart_rows[best_idx]["mean_confidence"]),
                        "auc_coverage": float(restart_rows[best_idx]["auc_coverage"]),
                    }
                )

        ranking_rows.sort(key=lambda r: (str(r["target"]), -float(r["closeness"])))
        write_csv(v_dir / "curve.csv", curve_rows)
        write_csv(v_dir / "closeness_ranking.csv", ranking_rows)

        # Figure 4 style line plot: coverage vs confidence threshold per target.
        fig_path = v_dir / "curve.png"
        targets = sorted({row["target"] for row in curve_rows})
        if targets:
            fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4), sharey=True)
            if len(targets) == 1:
                axes = [axes]
            for ax, target in zip(axes, targets):
                t_rows = [r for r in curve_rows if r["target"] == target]
                languages = sorted({r["candidate_language"] for r in t_rows})
                for language in languages:
                    l_rows = [r for r in t_rows if r["candidate_language"] == language]
                    l_rows.sort(key=lambda r: float(r["threshold"]))
                    xs = [float(r["threshold"]) for r in l_rows]
                    ys = [float(r["coverage"]) for r in l_rows]
                    ax.plot(xs, ys, label=language)
                ax.set_title(f"{target} closeness")
                ax.set_xlabel("confidence threshold")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("coverage")
            handles, labels = axes[-1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="center right", fontsize=8)
            fig.tight_layout(rect=(0, 0, 0.92, 1))
            fig.savefig(fig_path, dpi=180)
            plt.close(fig)

        write_json(
            v_dir / "metrics.json",
            {
                **now_meta(),
                "experiment": "iberian_closeness",
                "variant": variant.name,
                "restarts": restarts,
                "curve_csv": str(v_dir / "curve.csv"),
                "ranking_csv": str(v_dir / "closeness_ranking.csv"),
                "figure": str(fig_path),
            },
        )

    payload = {
        **now_meta(),
        "experiment": "iberian_closeness",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "variants": [v.name for v in variant_specs],
        "restarts": restarts,
    }
    write_json(out_dir / "run_summary.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit("Use python -m repro.run_experiment iberian-closeness")
