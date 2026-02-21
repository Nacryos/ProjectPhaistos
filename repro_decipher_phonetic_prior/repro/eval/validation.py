"""Validation-branch experiment runner.

Runs the phonetic-prior model against any language-family branch from the
ancient-scripts-datasets validation set.  Uses the registry's
``_build_validation_corpus()`` which provides concept-aligned cognate
pairs with IPA-first fallback.

Usage:
    python -m repro.run_experiment validation --branch germanic_expanded --smoke
    python -m repro.run_experiment validation --branch semitic --lost-lang pair:arabic:hebrew
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from datasets.registry import get_corpus
from repro.eval.common import (
    BilingualDataset,
    MissingDataError,
    TrainVariant,
    build_char_feature_matrix,
    check_sanity,
    compute_metrics,
    load_yaml,
    now_meta,
    rank_queries,
    ranking_records_to_rows,
    restart_seeds,
    summarize_restarts,
    train_model,
    write_csv,
    write_json,
    write_jsonl,
)


def _load_variants(cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[TrainVariant]:
    variants_cfg = cfg.get("variants", {})
    if not variants_cfg:
        raise ValueError("No variants defined in validation config.")
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


def _corpus_to_bilingual(corpus) -> BilingualDataset:
    """Convert a validation Corpus into a BilingualDataset for evaluation."""
    gold_map: Dict[str, List[str]] = {}
    for gt_row in (corpus.ground_truth or []):
        lost = gt_row.get("lost", "")
        known_list = gt_row.get("known", [])
        if isinstance(known_list, str):
            known_list = [known_list] if known_list else []
        if lost and known_list:
            gold_map.setdefault(lost, []).extend(known_list)
    # Deduplicate
    gold_map = {k: sorted(set(v)) for k, v in gold_map.items() if v}

    lost_queries = sorted(gold_map.keys())
    known_union = corpus.known_text.get("known_union", [])
    if not known_union:
        for vocab in corpus.known_text.values():
            known_union.extend(vocab)
        known_union = sorted(set(known_union))

    return BilingualDataset(
        name=corpus.name,
        lost_queries=lost_queries,
        gold_map=gold_map,
        known_vocab=known_union,
        metadata=dict(corpus.metadata),
    )


def run_validation(
    *,
    config_path: Path,
    output_root: Path,
    branch: str,
    lost_lang_variant: Optional[str],
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

    corpus_name = f"validation_{branch}"
    corpus = get_corpus(corpus_name, variant=lost_lang_variant)
    dataset = _corpus_to_bilingual(corpus)

    if not dataset.lost_queries:
        raise MissingDataError(f"No evaluation queries for validation branch {branch!r}.")
    if not dataset.known_vocab:
        raise MissingDataError(f"No known vocabulary for validation branch {branch!r}.")

    if max_queries > 0:
        kept = dataset.lost_queries[:max_queries]
        dataset = BilingualDataset(
            name=dataset.name,
            lost_queries=kept,
            gold_map={q: dataset.gold_map[q] for q in kept if q in dataset.gold_map},
            known_vocab=dataset.known_vocab,
            metadata=dict(dataset.metadata),
        )
    if smoke_known_vocab_max > 0 and len(dataset.known_vocab) > smoke_known_vocab_max:
        kept_vocab = set(dataset.known_vocab[:smoke_known_vocab_max])
        dataset = BilingualDataset(
            name=dataset.name,
            lost_queries=list(dataset.lost_queries),
            gold_map={q: [k for k in dataset.gold_map.get(q, []) if k in kept_vocab] for q in dataset.lost_queries},
            known_vocab=dataset.known_vocab[:smoke_known_vocab_max],
            metadata=dict(dataset.metadata),
        )

    train_text = corpus.lost_text
    if not train_text:
        raise MissingDataError(f"Validation branch {branch!r} produced empty lost text.")
    if smoke_train_lines_max > 0:
        train_text = train_text[:smoke_train_lines_max]

    out_dir = output_root / f"validation_{branch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(k) for k in cfg.get("eval_ks", [1, 3, 5, 10])]
    top_k = max(ks)
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
            metrics = compute_metrics(records, ks=ks)
            check_sanity(records)

            rows = ranking_records_to_rows(records, top_k=top_k)
            write_csv(run_dir / "per_query.csv", rows)
            write_jsonl(run_dir / "per_query.jsonl", rows)
            write_json(
                run_dir / "metrics.json",
                {
                    **now_meta(),
                    "experiment": f"validation_{branch}",
                    "variant": variant.name,
                    "seed": seed,
                    "metrics": metrics,
                    "history": train_out.history,
                    "dataset": dataset.metadata,
                },
            )

            row = {
                "seed": seed,
                **{f"p_at_{k}": float(metrics.get(f"p_at_{k}", 0.0)) for k in ks},
                "mrr": float(metrics.get("mrr", 0.0)),
            }
            restart_rows.append(row)
            restart_records.append(rows)

        primary_metric = f"p_at_{max(ks)}"
        summary = summarize_restarts(restart_rows, primary_metric=primary_metric)
        best_idx = next((i for i, r in enumerate(restart_rows) if r["seed"] == summary["best_seed"]), 0)
        best_rows = restart_records[best_idx]

        write_csv(v_dir / "per_query.csv", best_rows)
        write_jsonl(v_dir / "per_query.jsonl", best_rows)
        write_csv(v_dir / "restart_summary.csv", restart_rows)

        payload = {
            **now_meta(),
            "experiment": f"validation_{branch}",
            "variant": variant.name,
            "restarts": restarts,
            "seed_base": seed_base,
            "primary_metric": primary_metric,
            "summary": summary,
            "restart_rows": restart_rows,
            "config_path": str(config_path),
        }
        write_json(v_dir / "metrics.json", payload)
        variant_payloads[variant.name] = payload

        table_rows.append(
            {
                "branch": branch,
                "lost_language": corpus.metadata.get("lost_language", "?"),
                "known_languages": "|".join(corpus.metadata.get("known_languages", [])),
                "variant": variant.name,
                "metric": primary_metric,
                "score_best": float(summary["best"]),
                "score_mean": float(summary["mean"]),
                "score_std": float(summary["std"]),
                "best_seed": int(summary["best_seed"]),
            }
        )

    write_csv(out_dir / "results.csv", table_rows)

    payload = {
        **now_meta(),
        "experiment": f"validation_{branch}",
        "branch": branch,
        "lost_language": corpus.metadata.get("lost_language"),
        "known_languages": corpus.metadata.get("known_languages"),
        "available_languages": corpus.metadata.get("available_languages"),
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "rows": table_rows,
        "variants": list(variant_payloads.keys()),
        "restarts": restarts,
    }
    write_json(out_dir / "run_summary.json", payload)
    return payload


if __name__ == "__main__":
    raise SystemExit("Use python -m repro.run_experiment validation --branch <name>")
