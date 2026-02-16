"""Run phonetic-prior model on validation language branches/pairs."""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from datasets.registry import get_corpus, list_validation_languages
from repro.model import PhoneticPriorConfig, PhoneticPriorModel, train_one_step
from repro.paths import ARTIFACTS
from repro.utils import set_global_seeds, utc_now_iso, write_json


def _pick_variant(branch: str, lost: Optional[str], known: Optional[str], variant: Optional[str]) -> Optional[str]:
    if variant:
        return variant
    if lost and known:
        return f"pair:{lost.lower()}:{known.lower()}"
    if lost:
        return f"lang:{lost.lower()}"
    # Default to the first available language as lost; all others known.
    languages = list_validation_languages(f"validation_{branch}")
    if not languages:
        return None
    return f"lang:{languages[0]}"


def _known_vocab_from_corpus(corpus, known: Optional[str]) -> Tuple[List[str], str]:
    if known:
        known_key = known.lower()
        vocab = corpus.known_text.get(known_key)
        if not vocab:
            raise ValueError(f"Known language {known!r} not available in this corpus. Choices: {sorted(corpus.known_text)}")
        return sorted(set(vocab)), known_key

    if "known_union" in corpus.known_text:
        return sorted(set(corpus.known_text["known_union"])), "known_union"

    # Fallback to first known language key.
    first_key = sorted(corpus.known_text.keys())[0]
    return sorted(set(corpus.known_text[first_key])), first_key


def _rankdata_average(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _roc_auc(score_label_pairs: Sequence[Tuple[float, bool]]) -> float:
    if not score_label_pairs:
        return float("nan")
    scores = [x[0] for x in score_label_pairs]
    labels = [x[1] for x in score_label_pairs]
    n_pos = sum(1 for x in labels if x)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(scores)
    sum_pos_ranks = sum(rank for rank, is_pos in zip(ranks, labels) if is_pos)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _eval_pair_metrics(
    model: PhoneticPriorModel,
    lost_tokens: Sequence[str],
    known_vocab: Sequence[str],
    ground_truth: Sequence[Dict],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    char_distr = model.compute_char_distr().detach()
    examples: List[Dict[str, object]] = []

    correct_top1 = 0
    total_eval = 0
    fp_top1 = 0
    mrr_sum = 0.0

    scored_pairs: List[Tuple[float, bool]] = []
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    for token, gt in zip(lost_tokens, ground_truth):
        candidates = gt.get("known", [])
        if isinstance(candidates, str):
            candidates = [candidates]
        candidate_set = {str(c) for c in candidates if str(c)}
        concept_id = str(gt.get("concept_id", ""))

        scored: List[Tuple[float, str, bool]] = []
        for known in known_vocab:
            score = float(model.edit_distance_dp(token, known, char_distr).detach().cpu().item())
            is_pos = known in candidate_set
            scored.append((score, known, is_pos))
            scored_pairs.append((score, is_pos))
            if is_pos:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

        scored.sort(key=lambda x: x[0], reverse=True)
        pred = scored[0][1] if scored else ""
        top1_correct = bool(scored and scored[0][2])

        first_pos_rank = None
        for rank, (_, _, is_pos) in enumerate(scored, start=1):
            if is_pos:
                first_pos_rank = rank
                break

        if candidate_set:
            total_eval += 1
            correct_top1 += int(top1_correct)
            fp_top1 += int(not top1_correct)
            if first_pos_rank is not None:
                mrr_sum += 1.0 / float(first_pos_rank)

        examples.append(
            {
                "lost": token,
                "concept_id": concept_id,
                "pred_top1": pred,
                "gold_candidates": sorted(candidate_set),
                "correct": top1_correct,
                "score_top1": scored[0][0] if scored else None,
                "first_positive_rank": first_pos_rank,
            }
        )

    p_at_1 = (correct_top1 / total_eval) if total_eval else 0.0
    mrr = (mrr_sum / total_eval) if total_eval else 0.0
    auroc = _roc_auc(scored_pairs)
    mean_pos = (sum(pos_scores) / len(pos_scores)) if pos_scores else float("nan")
    mean_neg = (sum(neg_scores) / len(neg_scores)) if neg_scores else float("nan")
    score_margin = (mean_pos - mean_neg) if not (math.isnan(mean_pos) or math.isnan(mean_neg)) else float("nan")
    top1_fpr = (fp_top1 / total_eval) if total_eval else 0.0

    metrics = {
        "p_at_1": p_at_1,
        "mrr": mrr,
        "top1_false_positive_rate": top1_fpr,
        "auroc": auroc,
        "mean_positive_score": mean_pos,
        "mean_negative_score": mean_neg,
        "score_margin": score_margin,
        "n_eval": float(total_eval),
        "n_pos_scored": float(len(pos_scores)),
        "n_neg_scored": float(len(neg_scores)),
    }
    return metrics, examples


def run_validation(
    branch: str,
    lost: Optional[str],
    known: Optional[str],
    variant: Optional[str],
    steps: int,
    max_items: int,
    seed: int,
) -> Dict[str, object]:
    set_global_seeds(seed)

    resolved_variant = _pick_variant(branch=branch, lost=lost, known=known, variant=variant)
    corpus_name = f"validation_{branch.lower()}"
    corpus = get_corpus(corpus_name, variant=resolved_variant)
    known_vocab, known_key = _known_vocab_from_corpus(corpus, known=known)
    lost_text = corpus.lost_text[:max_items] if max_items > 0 else corpus.lost_text
    ground_truth = (corpus.ground_truth or [])[: len(lost_text)]
    if not lost_text:
        raise ValueError(f"No lost text found for corpus={corpus_name}, variant={resolved_variant}")
    if not known_vocab:
        raise ValueError(f"No known vocabulary found for corpus={corpus_name}, variant={resolved_variant}")

    chars_lost = sorted(set("".join(lost_text)))
    chars_known = sorted(set("".join(known_vocab)))
    cfg = PhoneticPriorConfig(
        temperature=0.2,
        alpha=3.5,
        lambda_cov=1.0,
        lambda_loss=1.0,
        min_span=1,
        max_span=8,
        embedding_dim=32,
        lr=0.15,
        seed=seed,
    )
    model = PhoneticPriorModel(chars_lost, chars_known, config=cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    history: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        out = train_one_step(model, optimizer, lost_text, known_vocab)
        history.append(
            {
                "step": step,
                "objective": out.objective,
                "quality": out.quality,
                "omega_cov": out.omega_cov,
                "omega_loss": out.omega_loss,
            }
        )

    metrics, examples = _eval_pair_metrics(model, lost_text, known_vocab, ground_truth)
    run_name = f"{branch.lower()}_{(corpus.metadata.get('lost_language') or 'lost')}_to_{known_key}"
    out_path = ARTIFACTS / "runs" / f"validation_{run_name}.json"
    result: Dict[str, object] = {
        "status": "ok",
        "created_at": utc_now_iso(),
        "corpus": corpus.name,
        "variant": corpus.variant,
        "branch": branch.lower(),
        "lost_language": corpus.metadata.get("lost_language"),
        "known_languages": corpus.metadata.get("known_languages"),
        "known_vocab_key": known_key,
        "num_lost_tokens": len(lost_text),
        "num_known_vocab": len(known_vocab),
        "steps": steps,
        "seed": seed,
        "p_at_1": metrics["p_at_1"],
        "mrr": metrics["mrr"],
        "top1_false_positive_rate": metrics["top1_false_positive_rate"],
        "auroc": metrics["auroc"],
        "mean_positive_score": metrics["mean_positive_score"],
        "mean_negative_score": metrics["mean_negative_score"],
        "score_margin": metrics["score_margin"],
        "n_eval": int(metrics["n_eval"]),
        "n_pos_scored": int(metrics["n_pos_scored"]),
        "n_neg_scored": int(metrics["n_neg_scored"]),
        "history": history,
        "examples": examples[:50],
        "metadata": corpus.metadata,
    }
    write_json(out_path, result)
    result["out_path"] = str(out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-language training through the phonetic-prior model.")
    parser.add_argument("--branch", required=True, help="Validation branch, e.g. germanic, semitic, celtic.")
    parser.add_argument("--lost", help="Lost language code (e.g., got, heb).")
    parser.add_argument("--known", help="Known language code for pair runs (e.g., ang, arb).")
    parser.add_argument(
        "--variant",
        help="Explicit dataset variant. Supports lang:<lost_lang> or pair:<lost_lang>:<known_lang>.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Training steps.")
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on number of lost tokens.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    result = run_validation(
        branch=args.branch,
        lost=args.lost,
        known=args.known,
        variant=args.variant,
        steps=args.steps,
        max_items=args.max_items,
        seed=args.seed,
    )
    print(f"Wrote {result['out_path']}")


if __name__ == "__main__":
    main()
