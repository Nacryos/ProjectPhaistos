"""Paper-style Gothic experiment (Table 2 + Table 4 style outputs)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from datasets.registry import get_corpus
from repro.eval.common import (
    BilingualDataset,
    MissingDataError,
    RankingRecord,
    TrainVariant,
    apply_whitespace_ratio,
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
from repro.paths import ROOT


DEFAULT_LOST_COL_CANDIDATES = [
    "gothic",
    "got",
    "lost",
    "source",
    "lost_stem",
    "gothic_stem",
    "query",
]
DEFAULT_KNOWN_COL_CANDIDATES = [
    "known",
    "target",
    "cognate",
    "stem",
    "known_stem",
    "proto_germanic",
    "old_norse",
    "old_english",
]


@dataclass
class SettingRun:
    restart_rows: List[Dict[str, Any]]
    best_records: List[RankingRecord]
    summary: Dict[str, Any]


def _load_variants(cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[TrainVariant]:
    variants_cfg = cfg.get("variants", {})
    if not variants_cfg:
        raise ValueError("No variants defined in Gothic config.")

    names = list(selected) if selected else list(variants_cfg.keys())
    variants: List[TrainVariant] = []
    for name in names:
        if name not in variants_cfg:
            raise ValueError(f"Unknown variant {name!r}. Available: {list(variants_cfg.keys())}")
        spec = variants_cfg[name]
        variants.append(
            TrainVariant(
                name=name,
                ipa=bool(spec.get("ipa", True)),
                omega_loss=bool(spec.get("omega_loss", True)),
                mapping_prior=str(spec.get("mapping_prior", "none")),
                partial_chars=[str(x).lower() for x in spec.get("partial_chars", [])],
                prior_strength=float(spec.get("prior_strength", 8.0)),
            )
        )
    return variants


def _load_known_sets(cfg: Mapping[str, Any], max_queries: int) -> Dict[str, BilingualDataset]:
    known_languages = cfg.get("known_languages", {})
    if not known_languages:
        raise ValueError("No known_languages section found in Gothic config.")

    datasets: Dict[str, BilingualDataset] = {}
    min_lost_len = int(cfg.get("min_lost_stem_len", 1))
    min_known_len = int(cfg.get("min_known_stem_len", 4))

    for label, spec in known_languages.items():
        path = resolve_path(str(spec.get("path", "")))
        lost_cols = spec.get("lost_col_candidates", DEFAULT_LOST_COL_CANDIDATES)
        known_cols = spec.get("known_col_candidates", DEFAULT_KNOWN_COL_CANDIDATES)

        dataset = load_bilingual_dataset(
            name=f"gothic_{label.lower()}",
            path=path,
            lost_col_candidates=[str(x) for x in lost_cols],
            known_col_candidates=[str(x) for x in known_cols],
            min_lost_len=min_lost_len,
            min_known_len=min_known_len,
        )
        if max_queries > 0:
            kept = dataset.lost_queries[:max_queries]
            dataset = BilingualDataset(
                name=dataset.name,
                lost_queries=kept,
                gold_map={q: dataset.gold_map[q] for q in kept},
                known_vocab=dataset.known_vocab,
                metadata=dict(dataset.metadata),
            )
        datasets[str(label)] = dataset

    return datasets


def _truncate_dataset_vocab(ds: BilingualDataset, max_vocab: int) -> BilingualDataset:
    if max_vocab <= 0 or len(ds.known_vocab) <= max_vocab:
        return ds
    kept_vocab = set(ds.known_vocab[:max_vocab])
    trimmed_gold: Dict[str, List[str]] = {}
    for q in ds.lost_queries:
        kept = [k for k in ds.gold_map.get(q, []) if k in kept_vocab]
        trimmed_gold[q] = kept
    return BilingualDataset(
        name=ds.name,
        lost_queries=list(ds.lost_queries),
        gold_map=trimmed_gold,
        known_vocab=ds.known_vocab[:max_vocab],
        metadata=dict(ds.metadata),
    )


def _run_setting(
    *,
    output_dir: Path,
    setting_name: str,
    seeds: Sequence[int],
    train_text: Sequence[str],
    dataset: BilingualDataset,
    variant: TrainVariant,
    train_cfg: Mapping[str, Any],
    top_k: int,
) -> SettingRun:
    restart_rows: List[Dict[str, Any]] = []
    restart_records: List[List[RankingRecord]] = []

    for restart_idx, seed in enumerate(seeds, start=1):
        run_dir = output_dir / "restarts" / f"{setting_name}_restart_{restart_idx:02d}"
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

        restart_payload = {
            **now_meta(),
            "setting": setting_name,
            "seed": seed,
            "variant": variant.name,
            "metrics": metrics,
            "history": train_out.history,
            "dataset": dataset.metadata,
        }
        write_json(run_dir / "metrics.json", restart_payload)
        rows = ranking_records_to_rows(records, top_k=top_k)
        write_csv(run_dir / "per_query.csv", rows)
        write_jsonl(run_dir / "per_query.jsonl", rows)

        row = {
            "setting": setting_name,
            "seed": seed,
            "p_at_1": float(metrics.get("p_at_1", 0.0)),
            "p_at_10": float(metrics.get("p_at_10", 0.0)),
            "mrr": float(metrics.get("mrr", 0.0)),
            "n_eval": int(metrics.get("n_eval", 0)),
            "metrics_path": str(run_dir / "metrics.json"),
            "per_query_path": str(run_dir / "per_query.csv"),
        }
        restart_rows.append(row)
        restart_records.append(records)

    summary = summarize_restarts(restart_rows, primary_metric="p_at_10")
    best_seed = summary.get("best_seed")
    best_idx = next((i for i, r in enumerate(restart_rows) if r["seed"] == best_seed), 0)
    best_records = restart_records[best_idx]

    return SettingRun(restart_rows=restart_rows, best_records=best_records, summary=summary)


def run_gothic(
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
    top_k = int(cfg.get("top_k", 10))
    whitespace_ratios = [int(x) for x in cfg.get("whitespace_ratios", [0, 25, 50, 75])]
    known_datasets = _load_known_sets(cfg, max_queries=max_queries)

    if smoke:
        restarts = 1
        max_queries = 50 if max_queries <= 0 else min(max_queries, 50)
        train_cfg = dict(train_cfg)
        train_cfg["num_steps"] = int(cfg.get("smoke_num_steps", 20))
        smoke_known_vocab_max = int(cfg.get("smoke_known_vocab_max", 200))
        known_datasets = {
            key: _truncate_dataset_vocab(ds, smoke_known_vocab_max)
            for key, ds in known_datasets.items()
        }
        smoke_train_lines_max = int(cfg.get("smoke_train_lines_max", 32))
    else:
        smoke_train_lines_max = 0

    variant_specs = _load_variants(cfg, variants)

    gothic_corpus = get_corpus("gothic")
    base_train_lines = gothic_corpus.lost_text
    if not base_train_lines:
        raise MissingDataError("Gothic corpus is empty in datasets registry.")

    if max_queries > 0:
        train_limit = int(cfg.get("max_training_lines", max_queries * 8))
        base_train_lines = base_train_lines[:train_limit]
    if smoke_train_lines_max > 0:
        base_train_lines = base_train_lines[:smoke_train_lines_max]

    out_dir = output_root / "gothic"
    out_dir.mkdir(parents=True, exist_ok=True)

    table2_rows: List[Dict[str, Any]] = []
    variant_summaries: Dict[str, Any] = {}

    for variant in variant_specs:
        v_dir = out_dir / variant.name
        v_dir.mkdir(parents=True, exist_ok=True)

        best_per_query_rows: List[Dict[str, Any]] = []
        setting_rows: List[Dict[str, Any]] = []

        for wr in whitespace_ratios:
            for known_label, dataset in known_datasets.items():
                setting = f"wr{wr}_{known_label}"
                seeds = restart_seeds(seed_base + wr * 100 + len(known_label), restarts)
                train_text = apply_whitespace_ratio(base_train_lines, wr, seed=seeds[0])

                run = _run_setting(
                    output_dir=v_dir,
                    setting_name=setting,
                    seeds=seeds,
                    train_text=train_text,
                    dataset=dataset,
                    variant=variant,
                    train_cfg=train_cfg,
                    top_k=top_k,
                )

                summary = run.summary
                setting_rows.append(
                    {
                        "whitespace_ratio": wr,
                        "known_language": known_label,
                        "variant": variant.name,
                        "p_at_10_best": float(summary["best"]),
                        "p_at_10_mean": float(summary["mean"]),
                        "p_at_10_std": float(summary["std"]),
                        "best_seed": int(summary["best_seed"]),
                    }
                )

                for row in ranking_records_to_rows(run.best_records, top_k=top_k):
                    row = dict(row)
                    row["whitespace_ratio"] = wr
                    row["known_language"] = known_label
                    row["variant"] = variant.name
                    best_per_query_rows.append(row)

        write_csv(v_dir / "per_query.csv", best_per_query_rows)
        write_jsonl(v_dir / "per_query.jsonl", best_per_query_rows)
        write_csv(v_dir / "restart_summary.csv", setting_rows)

        metrics_payload = {
            **now_meta(),
            "experiment": "gothic",
            "variant": variant.name,
            "restarts": restarts,
            "seed_base": seed_base,
            "top_k": top_k,
            "settings": setting_rows,
            "config_path": str(config_path),
        }
        write_json(v_dir / "metrics.json", metrics_payload)
        variant_summaries[variant.name] = metrics_payload

    key_order = ["base", "partial", "full"]
    rows_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for variant_name in key_order:
        v_dir = out_dir / variant_name
        if not v_dir.exists():
            continue
        summary_csv = v_dir / "restart_summary.csv"
        if not summary_csv.exists():
            continue
        with summary_csv.open("r", encoding="utf8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                wr = int(float(row["whitespace_ratio"]))
                known = row["known_language"]
                key = (wr, known)
                rows_by_key.setdefault(key, {"whitespace_ratio": wr, "known_language": known})
                rows_by_key[key][variant_name] = float(row["p_at_10_best"])

    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        table2_rows.append(
            {
                "whitespace_ratio": row["whitespace_ratio"],
                "known_language": row["known_language"],
                "base": row.get("base"),
                "partial": row.get("partial"),
                "full": row.get("full"),
            }
        )

    write_csv(out_dir / "table2.csv", table2_rows)

    ablation_cfg = cfg.get("ablation", {})
    table4_rows: List[Dict[str, Any]] = []
    if ablation_cfg:
        target_wr = int(ablation_cfg.get("whitespace_ratio", 75))
        target_known = str(ablation_cfg.get("known_language", "ON"))
        ablations = ablation_cfg.get(
            "rows",
            [
                {"ipa": True, "omega_loss": True, "label": "ipa_on_omega_on"},
                {"ipa": False, "omega_loss": True, "label": "ipa_off_omega_on"},
                {"ipa": True, "omega_loss": False, "label": "ipa_on_omega_off"},
            ],
        )

        if target_known not in known_datasets:
            raise MissingDataError(
                f"Ablation target known language {target_known!r} is not in Gothic known_languages config."
            )

        for row_cfg in ablations:
            row_variant_scores: Dict[str, float] = {}
            for base_name in ["base", "partial", "full"]:
                if base_name not in [v.name for v in variant_specs]:
                    continue
                base_variant = next(v for v in variant_specs if v.name == base_name)
                custom_variant = TrainVariant(
                    name=f"{base_name}_{row_cfg.get('label', 'ablation')}",
                    ipa=bool(row_cfg.get("ipa", base_variant.ipa)),
                    omega_loss=bool(row_cfg.get("omega_loss", base_variant.omega_loss)),
                    mapping_prior=base_variant.mapping_prior,
                    partial_chars=base_variant.partial_chars,
                    prior_strength=base_variant.prior_strength,
                )

                seeds = restart_seeds(seed_base + target_wr * 100 + len(target_known) + 5000, restarts)
                train_text = apply_whitespace_ratio(base_train_lines, target_wr, seed=seeds[0])
                run = _run_setting(
                    output_dir=out_dir / "ablations",
                    setting_name=f"{row_cfg.get('label', 'row')}_wr{target_wr}_{target_known}_{base_name}",
                    seeds=seeds,
                    train_text=train_text,
                    dataset=known_datasets[target_known],
                    variant=custom_variant,
                    train_cfg=train_cfg,
                    top_k=top_k,
                )
                row_variant_scores[base_name] = float(run.summary["best"])

            table4_rows.append(
                {
                    "ipa": bool(row_cfg.get("ipa", True)),
                    "omega_loss": bool(row_cfg.get("omega_loss", True)),
                    "base": row_variant_scores.get("base"),
                    "partial": row_variant_scores.get("partial"),
                    "full": row_variant_scores.get("full"),
                }
            )

        write_csv(out_dir / "table4.csv", table4_rows)

    payload = {
        **now_meta(),
        "experiment": "gothic",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "table2_rows": table2_rows,
        "table4_rows": table4_rows,
        "variants": list(variant_summaries.keys()),
        "restarts": restarts,
    }
    write_json(out_dir / "run_summary.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit("Use python -m repro.run_experiment gothic")
