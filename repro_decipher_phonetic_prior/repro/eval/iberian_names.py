"""Paper-style Iberian personal-names evaluation (Figure 4a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt

from datasets.registry import get_corpus
from repro.eval.visualize import save_char_distr
from repro.eval.common import (
    MissingDataError,
    TrainVariant,
    check_sanity,
    compute_metrics,
    load_bilingual_dataset,
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


DEFAULT_LOST_COLS = ["iberian", "iberian_stem", "lost", "source", "lost_stem", "query"]
DEFAULT_KNOWN_COLS = ["latin", "latin_stem", "known", "target", "known_stem", "cognate"]


def _load_variants(cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[TrainVariant]:
    variants_cfg = cfg.get("variants", {})
    if not variants_cfg:
        raise ValueError("No variants defined in Iberian config.")
    names = list(selected) if selected else list(variants_cfg.keys())
    out: List[TrainVariant] = []
    for name in names:
        spec = variants_cfg.get(name)
        if spec is None:
            raise ValueError(f"Unknown Iberian variant {name!r}")
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


def run_iberian_names(
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

    names_path = resolve_path(str(cfg.get("personal_names_path", "")))
    names_ds = load_bilingual_dataset(
        name="iberian_latin_names",
        path=names_path,
        lost_col_candidates=[str(x) for x in cfg.get("name_lost_col_candidates", DEFAULT_LOST_COLS)],
        known_col_candidates=[str(x) for x in cfg.get("name_known_col_candidates", DEFAULT_KNOWN_COLS)],
        min_lost_len=int(cfg.get("min_lost_stem_len", 1)),
        min_known_len=int(cfg.get("min_known_stem_len", 2)),
    )

    if max_queries > 0:
        kept = names_ds.lost_queries[:max_queries]
        names_ds = type(names_ds)(
            name=names_ds.name,
            lost_queries=kept,
            gold_map={q: names_ds.gold_map[q] for q in kept},
            known_vocab=names_ds.known_vocab,
            metadata=dict(names_ds.metadata),
        )
    if smoke_known_vocab_max > 0 and len(names_ds.known_vocab) > smoke_known_vocab_max:
        kept_vocab = set(names_ds.known_vocab[:smoke_known_vocab_max])
        names_ds = type(names_ds)(
            name=names_ds.name,
            lost_queries=list(names_ds.lost_queries),
            gold_map={q: [k for k in names_ds.gold_map.get(q, []) if k in kept_vocab] for q in names_ds.lost_queries},
            known_vocab=names_ds.known_vocab[:smoke_known_vocab_max],
            metadata=dict(names_ds.metadata),
        )

    corpus = get_corpus("iberian", variant=corpus_variant)
    train_text = corpus.lost_text
    if not train_text:
        raise MissingDataError("Iberian corpus is empty in dataset registry.")
    if max_queries > 0:
        train_text = train_text[: max_queries * 8]
    if smoke_train_lines_max > 0:
        train_text = train_text[:smoke_train_lines_max]

    out_dir = output_root / "iberian_names"
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(k) for k in cfg.get("eval_ks", [1, 3, 5, 10])]
    top_k = max(ks)
    variant_specs = _load_variants(cfg, variants)

    p_at_k_rows: List[Dict[str, Any]] = []

    for variant in variant_specs:
        v_dir = out_dir / variant.name
        v_dir.mkdir(parents=True, exist_ok=True)

        restart_rows: List[Dict[str, Any]] = []
        restart_query_rows: List[List[Dict[str, Any]]] = []

        for restart_idx, seed in enumerate(restart_seeds(seed_base, restarts), start=1):
            run_dir = v_dir / "restarts" / f"restart_{restart_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_out = train_model(
                lost_training_text=train_text,
                known_vocab=names_ds.known_vocab,
                variant=variant,
                seed=seed,
                train_cfg=train_cfg,
            )
            save_char_distr(train_out.model, run_dir)
            records = rank_queries(
                model=train_out.model,
                queries=names_ds.lost_queries,
                gold_map=names_ds.gold_map,
                known_vocab=names_ds.known_vocab,
                top_k=top_k,
            )
            metrics = compute_metrics(records, ks=ks)
            check_sanity(records)

            rows = ranking_records_to_rows(records, top_k=top_k)
            write_csv(run_dir / "per_query.csv", rows)
            write_jsonl(run_dir / "per_query.jsonl", rows)
            write_json(
                run_dir / "metrics.json",
                {
                    **now_meta(),
                    "experiment": "iberian_names",
                    "variant": variant.name,
                    "seed": seed,
                    "metrics": metrics,
                    "history": train_out.history,
                    "dataset": names_ds.metadata,
                },
            )

            restart_rows.append({"seed": seed, **{f"p_at_{k}": float(metrics.get(f"p_at_{k}", 0.0)) for k in ks}, "mrr": float(metrics.get("mrr", 0.0))})
            restart_query_rows.append(rows)

        primary_metric = f"p_at_{max(ks)}"
        summary = summarize_restarts(restart_rows, primary_metric=primary_metric)
        best_idx = next((i for i, r in enumerate(restart_rows) if r["seed"] == summary["best_seed"]), 0)
        best_rows = restart_query_rows[best_idx]

        write_csv(v_dir / "per_query.csv", best_rows)
        write_jsonl(v_dir / "per_query.jsonl", best_rows)
        write_csv(v_dir / "restart_summary.csv", restart_rows)

        payload = {
            **now_meta(),
            "experiment": "iberian_names",
            "variant": variant.name,
            "restarts": restarts,
            "seed_base": seed_base,
            "primary_metric": primary_metric,
            "summary": summary,
            "restart_rows": restart_rows,
            "config_path": str(config_path),
        }
        write_json(v_dir / "metrics.json", payload)

        for k in ks:
            vals = [float(row[f"p_at_{k}"]) for row in restart_rows]
            p_at_k_rows.append(
                {
                    "variant": variant.name,
                    "k": k,
                    "p_at_k_best": float(max(vals) if vals else 0.0),
                    "p_at_k_mean": float(sum(vals) / len(vals) if vals else 0.0),
                }
            )

    write_csv(out_dir / "p_at_k.csv", p_at_k_rows)

    # Plot figure 4a style P@K curve.
    fig_path = out_dir / "p_at_k_curve.png"
    if p_at_k_rows:
        plt.figure(figsize=(6, 4))
        variant_names = sorted(set(row["variant"] for row in p_at_k_rows))
        for variant_name in variant_names:
            rows = [r for r in p_at_k_rows if r["variant"] == variant_name]
            rows.sort(key=lambda r: int(r["k"]))
            ks_sorted = [int(r["k"]) for r in rows]
            vals = [float(r["p_at_k_best"]) for r in rows]
            plt.plot(ks_sorted, vals, marker="o", label=variant_name)
        plt.xlabel("K")
        plt.ylabel("P@K")
        plt.title("Iberian Personal Names (Figure 4a style)")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()

    payload = {
        **now_meta(),
        "experiment": "iberian_names",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "p_at_k_rows": p_at_k_rows,
        "figure": str(fig_path),
        "restarts": restarts,
    }
    write_json(out_dir / "run_summary.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit("Use python -m repro.run_experiment iberian-names")
