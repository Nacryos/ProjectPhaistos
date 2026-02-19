"""Paper-style Ugaritic experiment (P@1, Table 3 style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from datasets.registry import get_corpus
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
from repro.reference.paper_metrics import TABLE3


DEFAULT_LOST_COLS = ["uga-no_spe", "ugaritic", "lost", "source", "lost_stem"]
DEFAULT_KNOWN_COLS = ["heb-no_spe", "hebrew", "known", "target", "known_stem"]


def _load_variants(cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[TrainVariant]:
    variants_cfg = cfg.get("variants", {})
    if not variants_cfg:
        raise ValueError("No variants defined in Ugaritic config.")
    names = list(selected) if selected else list(variants_cfg.keys())
    out: List[TrainVariant] = []
    for name in names:
        spec = variants_cfg.get(name)
        if spec is None:
            raise ValueError(f"Unknown Ugaritic variant {name!r}")
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


def run_ugaritic(
    *,
    config_path: Path,
    output_root: Path,
    variants: Optional[Sequence[str]],
    restarts: int,
    seed_base: int,
    max_queries: int,
    smoke: bool,
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

    path = resolve_path(str(cfg.get("cognate_path", "")))
    dataset = load_bilingual_dataset(
        name="ugaritic_hebrew",
        path=path,
        lost_col_candidates=[str(x) for x in cfg.get("lost_col_candidates", DEFAULT_LOST_COLS)],
        known_col_candidates=[str(x) for x in cfg.get("known_col_candidates", DEFAULT_KNOWN_COLS)],
        min_lost_len=int(cfg.get("min_lost_stem_len", 1)),
        min_known_len=int(cfg.get("min_known_stem_len", 3)),
    )

    if max_queries > 0:
        kept = dataset.lost_queries[:max_queries]
        dataset = type(dataset)(
            name=dataset.name,
            lost_queries=kept,
            gold_map={q: dataset.gold_map[q] for q in kept},
            known_vocab=dataset.known_vocab,
            metadata=dict(dataset.metadata),
        )
    if smoke_known_vocab_max > 0 and len(dataset.known_vocab) > smoke_known_vocab_max:
        kept_vocab = set(dataset.known_vocab[:smoke_known_vocab_max])
        dataset = type(dataset)(
            name=dataset.name,
            lost_queries=list(dataset.lost_queries),
            gold_map={q: [k for k in dataset.gold_map.get(q, []) if k in kept_vocab] for q in dataset.lost_queries},
            known_vocab=dataset.known_vocab[:smoke_known_vocab_max],
            metadata=dict(dataset.metadata),
        )

    corpus = get_corpus("ugaritic")
    train_text = corpus.lost_text
    if not train_text:
        raise MissingDataError("Ugaritic corpus is empty in dataset registry.")
    if max_queries > 0:
        train_text = train_text[: max_queries * 8]
    if smoke_train_lines_max > 0:
        train_text = train_text[:smoke_train_lines_max]

    out_dir = output_root / "ugaritic"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(cfg.get("top_k", 10))
    variant_specs = _load_variants(cfg, variants)

    table_rows: List[Dict[str, Any]] = []
    variant_payloads: Dict[str, Any] = {}

    for variant in variant_specs:
        v_dir = out_dir / variant.name
        v_dir.mkdir(parents=True, exist_ok=True)

        restart_rows: List[Dict[str, Any]] = []
        restart_records: List[List[Dict[str, Any]]] = []

        for restart_idx, seed in enumerate(restart_seeds(seed_base, restarts), start=1):
            run_dir = v_dir / "restarts" / f"restart_{restart_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_out = train_model(
                lost_training_text=train_text,
                known_vocab=dataset.known_vocab,
                variant=variant,
                seed=seed,
                train_cfg=train_cfg,
            )
            records = rank_queries(
                model=train_out.model,
                queries=dataset.lost_queries,
                gold_map=dataset.gold_map,
                known_vocab=dataset.known_vocab,
                top_k=top_k,
            )
            metrics = compute_metrics(records, ks=[1, 10])
            check_sanity(records)

            rows = ranking_records_to_rows(records, top_k=top_k)
            write_csv(run_dir / "per_query.csv", rows)
            write_jsonl(run_dir / "per_query.jsonl", rows)
            write_json(
                run_dir / "metrics.json",
                {
                    **now_meta(),
                    "experiment": "ugaritic",
                    "variant": variant.name,
                    "seed": seed,
                    "metrics": metrics,
                    "history": train_out.history,
                    "dataset": dataset.metadata,
                },
            )

            restart_rows.append(
                {
                    "seed": seed,
                    "p_at_1": float(metrics.get("p_at_1", 0.0)),
                    "p_at_10": float(metrics.get("p_at_10", 0.0)),
                    "mrr": float(metrics.get("mrr", 0.0)),
                }
            )
            restart_records.append(rows)

        summary = summarize_restarts(restart_rows, primary_metric="p_at_1")
        best_idx = next((i for i, r in enumerate(restart_rows) if r["seed"] == summary["best_seed"]), 0)
        best_rows = restart_records[best_idx]

        write_csv(v_dir / "per_query.csv", best_rows)
        write_jsonl(v_dir / "per_query.jsonl", best_rows)
        write_csv(v_dir / "restart_summary.csv", restart_rows)

        payload = {
            **now_meta(),
            "experiment": "ugaritic",
            "variant": variant.name,
            "restarts": restarts,
            "seed_base": seed_base,
            "summary": summary,
            "restart_rows": restart_rows,
            "config_path": str(config_path),
        }
        write_json(v_dir / "metrics.json", payload)
        variant_payloads[variant.name] = payload

        table_rows.append(
            {
                "lost": "Ugaritic",
                "known": "Hebrew",
                "method": variant.name,
                "metric": "P@1",
                "score_best": float(summary["best"]),
                "score_mean": float(summary["mean"]),
                "score_std": float(summary["std"]),
                "best_seed": int(summary["best_seed"]),
            }
        )

    # Add comparison rows from paper baselines for table convenience.
    for row in TABLE3:
        if row.get("lost") != "Ugaritic":
            continue
        if row.get("method") in {"base", "partial", "full"}:
            continue
        table_rows.append(
            {
                "lost": row["lost"],
                "known": row["known"],
                "method": row["method"],
                "metric": row["metric"],
                "score_best": row["score"],
                "score_mean": row["score"],
                "score_std": 0.0,
                "best_seed": "paper",
            }
        )

    write_csv(out_dir / "table3_ugaritic.csv", table_rows)

    payload = {
        **now_meta(),
        "experiment": "ugaritic",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "rows": table_rows,
        "variants": list(variant_payloads.keys()),
        "restarts": restarts,
    }
    write_json(out_dir / "run_summary.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit("Use python -m repro.run_experiment ugaritic")
