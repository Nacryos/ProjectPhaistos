"""Shared utilities for paper-style evaluation experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from repro.model import PhoneticPriorConfig, PhoneticPriorModel, train_one_step
from repro.paths import ROOT
from repro.utils import set_global_seeds, utc_now_iso


_TOKEN_RE = re.compile(r"[^\wþðƕȝōēæáéíóúïöüãẽĩõũȳȧḱǫśžčşẓḫʻʿ`@\-]+", flags=re.UNICODE)


class MissingDataError(RuntimeError):
    """Raised when required data files for paper runs are unavailable."""


@dataclass
class BilingualDataset:
    name: str
    lost_queries: List[str]
    gold_map: Dict[str, List[str]]
    known_vocab: List[str]
    metadata: Dict[str, Any]


@dataclass
class RankingRecord:
    query_id: int
    query: str
    gold_candidates: List[str]
    top_predictions: List[str]
    top_scores: List[float]
    first_positive_rank: Optional[int]
    top1_confidence: float


@dataclass
class TrainVariant:
    name: str
    ipa: bool
    omega_loss: bool
    mapping_prior: str  # none|partial|full
    partial_chars: List[str]
    prior_strength: float


@dataclass
class TrainResult:
    model: PhoneticPriorModel
    history: List[Dict[str, float]]


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf8"))


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p


def ensure_file(path: Path, note: str) -> None:
    if not path.exists():
        raise MissingDataError(
            f"Missing required file for {note}: {path}\n"
            f"Create/download this file and re-run. See README data acquisition note."
        )


def normalize_token(text: str) -> str:
    cleaned = _TOKEN_RE.sub(" ", text.strip().lower())
    parts = [p for p in cleaned.split() if p]
    return " ".join(parts)


def split_multi(text: str) -> List[str]:
    items = re.split(r"[|;,]", text)
    out = []
    for it in items:
        tok = normalize_token(it)
        if tok:
            out.append(tok)
    return out


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pick_column(fieldnames: Sequence[str], preferred: Sequence[str], role: str, path: Path) -> str:
    if not fieldnames:
        raise MissingDataError(f"No header found in {path} while selecting column for {role}.")
    canon = {_canonical(col): col for col in fieldnames}
    for candidate in preferred:
        key = _canonical(candidate)
        if key in canon:
            return canon[key]
    raise MissingDataError(
        f"Could not find {role} column in {path}.\n"
        f"Available columns: {list(fieldnames)}\n"
        f"Expected one of: {list(preferred)}"
    )


def _read_rows(path: Path) -> List[Dict[str, str]]:
    ensure_file(path, "table parsing")
    with path.open("r", encoding="utf8") as fin:
        first = fin.readline()
        if not first:
            return []
        delim = "\t" if "\t" in first else ","
    with path.open("r", encoding="utf8") as fin:
        reader = csv.DictReader(fin, delimiter=delim)
        rows = [dict(row) for row in reader]
    return rows


def load_bilingual_dataset(
    *,
    name: str,
    path: Path,
    lost_col_candidates: Sequence[str],
    known_col_candidates: Sequence[str],
    min_lost_len: int,
    min_known_len: int,
) -> BilingualDataset:
    rows = _read_rows(path)
    if not rows:
        raise MissingDataError(f"No rows found in bilingual file: {path}")

    fieldnames = list(rows[0].keys())
    lost_col = _pick_column(fieldnames, lost_col_candidates, "lost", path)
    known_col = _pick_column(fieldnames, known_col_candidates, "known", path)

    gold_map: Dict[str, set[str]] = {}
    known_vocab_set: set[str] = set()

    for row in rows:
        lost_raw = row.get(lost_col, "") or ""
        known_raw = row.get(known_col, "") or ""
        # Skip placeholder rows (_.cog convention for missing cognates)
        if lost_raw.strip() == "_" or known_raw.strip() == "_":
            continue
        lost_tok = normalize_token(lost_raw)
        if not lost_tok:
            continue
        if len(lost_tok.replace(" ", "")) < min_lost_len:
            continue
        known_list = split_multi(known_raw)
        kept = [k for k in known_list if len(k.replace(" ", "")) >= min_known_len]
        if not kept:
            continue

        if lost_tok not in gold_map:
            gold_map[lost_tok] = set()
        for known_tok in kept:
            gold_map[lost_tok].add(known_tok)
            known_vocab_set.add(known_tok)

    lost_queries = sorted(gold_map)
    known_vocab = sorted(known_vocab_set)

    if not lost_queries:
        raise MissingDataError(f"No usable lost-side queries parsed from {path}")
    if not known_vocab:
        raise MissingDataError(f"No usable known-side vocabulary parsed from {path}")

    return BilingualDataset(
        name=name,
        lost_queries=lost_queries,
        gold_map={k: sorted(v) for k, v in gold_map.items()},
        known_vocab=known_vocab,
        metadata={
            "path": str(path),
            "lost_column": lost_col,
            "known_column": known_col,
            "num_rows": len(rows),
            "num_queries": len(lost_queries),
            "num_known_vocab": len(known_vocab),
        },
    )


def load_plain_vocab(path: Path, min_len: int) -> List[str]:
    ensure_file(path, "plain vocabulary")
    vocab: set[str] = set()
    for line in path.read_text(encoding="utf8").splitlines():
        tok = normalize_token(line)
        if tok and len(tok.replace(" ", "")) >= min_len:
            vocab.add(tok)
    if not vocab:
        raise MissingDataError(f"No usable vocabulary found in {path}")
    return sorted(vocab)


def load_vocab_from_xml(path: Path, min_len: int, max_items: int) -> List[str]:
    ensure_file(path, "XML vocabulary extraction")
    text = path.read_text(encoding="utf8", errors="ignore")
    segs = re.findall(r">([^<>]+)<", text)
    vocab: List[str] = []
    seen: set[str] = set()
    for seg in segs:
        clean = normalize_token(seg)
        for tok in clean.split():
            if len(tok) < min_len:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            vocab.append(tok)
            if max_items > 0 and len(vocab) >= max_items:
                return vocab
    if not vocab:
        raise MissingDataError(f"No usable tokens extracted from XML file {path}")
    return vocab


def apply_whitespace_ratio(lines: Sequence[str], ratio_percent: int, seed: int) -> List[str]:
    ratio = max(0.0, min(100.0, float(ratio_percent))) / 100.0
    if ratio >= 0.999:
        return [normalize_token(x) for x in lines if normalize_token(x)]

    rng = np.random.default_rng(seed)
    out: List[str] = []
    for raw in lines:
        line = normalize_token(raw)
        if not line:
            continue
        chars: List[str] = []
        for ch in line:
            if ch == " ":
                if rng.random() <= ratio:
                    chars.append(ch)
            else:
                chars.append(ch)
        collapsed = re.sub(r"\s+", " ", "".join(chars)).strip()
        if collapsed:
            out.append(collapsed)
    return out


# ---------------------------------------------------------------------------
# Panphon-based phonological feature system (Solution D: grouped embeddings)
# ---------------------------------------------------------------------------
# Feature groups matching standard phonological categories.  Each group is
# projected independently by the model so that sound-change dimensions
# (voicing, place, manner, …) live in orthogonal embedding subspaces.
# ---------------------------------------------------------------------------

# panphon feature names, grouped by phonological category.
PANPHON_FEATURE_GROUPS: Dict[str, List[str]] = {
    "major_class": ["syl", "son", "cons"],                       # 3
    "manner":      ["cont", "delrel", "lat", "nas", "strid"],    # 5
    "laryngeal":   ["voi", "sg", "cg"],                          # 3
    "place":       ["ant", "cor", "distr", "lab"],                # 4
    "vowel_body":  ["hi", "lo", "back", "round", "tense", "long"],  # 6
}
PANPHON_GROUP_ORDER: List[str] = list(PANPHON_FEATURE_GROUPS.keys())
PANPHON_GROUP_SIZES: List[int] = [len(PANPHON_FEATURE_GROUPS[g]) for g in PANPHON_GROUP_ORDER]

# Character -> IPA normalisations for symbols commonly found in our data
# that panphon does not recognise under their original codepoints.
_IPA_CHAR_NORM: Dict[str, str] = {
    "g": "\u0261",   # ASCII g -> IPA ɡ (U+0261)
    "þ": "θ",        # Gothic thorn -> voiceless dental fricative
    "ƕ": "xʷ",       # Gothic hwair -> voiceless velar fricative (labialized; use base)
    "ȝ": "ɣ",        # yogh -> voiced velar fricative
    "ō": "oː",       # macron vowels -> IPA long (base char used for features)
    "ē": "eː",
    "ā": "aː",
    "ī": "iː",
    "ū": "uː",
}

_panphon_ft: Any = None  # lazily initialised FeatureTable


def _get_panphon_ft() -> Any:
    """Return a cached panphon FeatureTable (import is deferred)."""
    global _panphon_ft
    if _panphon_ft is None:
        import os
        os.environ.setdefault("PYTHONUTF8", "1")
        import panphon  # type: ignore[import-untyped]
        _panphon_ft = panphon.FeatureTable()
    return _panphon_ft


def _ipa_char_to_vec(ch: str) -> List[float]:
    """Map a single IPA character to a panphon feature vector.

    Returns a flat list of floats (values in {-1, 0, +1}) whose length
    equals the sum of all group sizes (21 features).  Unknown characters
    get an all-zeros vector.
    """
    ft = _get_panphon_ft()
    # Normalise common ASCII fallbacks
    lookup = _IPA_CHAR_NORM.get(ch, ch)
    vecs = ft.word_to_vector_list(lookup, numeric=True)
    if vecs:
        # panphon may return >21 features in newer versions; take the
        # features we care about by name.
        full_vec = dict(zip(ft.names, vecs[0]))
        out: List[float] = []
        for group_name in PANPHON_GROUP_ORDER:
            for feat_name in PANPHON_FEATURE_GROUPS[group_name]:
                out.append(float(full_vec.get(feat_name, 0)))
        return out
    # Fallback: try stripping combining diacritics and retrying base char
    import unicodedata
    base = unicodedata.normalize("NFD", lookup)
    if base and base[0] != lookup:
        return _ipa_char_to_vec(base[0])
    # Unknown character: zero vector
    total_feats = sum(PANPHON_GROUP_SIZES)
    return [0.0] * total_feats


def build_char_feature_matrix(chars: Sequence[str], use_ipa_geometry: bool) -> torch.Tensor:
    """Build a phonological feature matrix for a character inventory.

    When *use_ipa_geometry* is True, returns a (num_chars, 21) tensor of
    panphon articulatory features grouped as:
        [major_class(3) | manner(5) | laryngeal(3) | place(4) | vowel_body(6)]

    Each group will be projected independently by the model (see
    ``GroupedIPAProjector`` in phonetic_prior.py) so that sound-change
    dimensions live in orthogonal embedding subspaces.

    When *use_ipa_geometry* is False, returns a one-hot identity matrix
    (no phonological structure).
    """
    chars = list(chars)
    if not chars:
        chars = ["?"]

    if not use_ipa_geometry:
        return torch.eye(len(chars), dtype=torch.float32)

    rows: List[List[float]] = []
    for ch in chars:
        rows.append(_ipa_char_to_vec(ch))
    return torch.tensor(rows, dtype=torch.float32)


def _apply_mapping_prior(model: PhoneticPriorModel, variant: TrainVariant) -> None:
    if variant.mapping_prior == "none":
        return

    selected: set[str] = set()
    if variant.mapping_prior == "full":
        selected = set(model.lost_chars) & set(model.known_chars)
    elif variant.mapping_prior == "partial":
        selected = {c.lower() for c in variant.partial_chars}
    else:
        raise ValueError(f"Unknown mapping_prior {variant.mapping_prior!r}")

    if not selected:
        return

    with torch.no_grad():
        for ch in selected:
            if ch not in model.lost2idx or ch not in model.known2idx:
                continue
            l_idx = model.lost2idx[ch]
            k_idx = model.known2idx[ch]
            model.mapping_logits[k_idx, l_idx] += float(variant.prior_strength)


def _annealed_alpha(alpha_start: float, alpha_end: float, anneal_steps: int, step: int) -> float:
    if anneal_steps <= 0:
        return alpha_end
    if step >= anneal_steps:
        return alpha_end
    ratio = float(step) / float(max(1, anneal_steps))
    return alpha_start + (alpha_end - alpha_start) * ratio


def train_model(
    *,
    lost_training_text: Sequence[str],
    known_vocab: Sequence[str],
    variant: TrainVariant,
    seed: int,
    train_cfg: Mapping[str, Any],
) -> TrainResult:
    set_global_seeds(seed)

    if not lost_training_text:
        raise ValueError("lost_training_text is empty")
    if not known_vocab:
        raise ValueError("known_vocab is empty")

    chars_lost = sorted(set("".join(lost_training_text)))
    chars_known = sorted(set("".join(known_vocab)))
    known_feats = build_char_feature_matrix(chars_known, use_ipa_geometry=variant.ipa)

    lambda_loss = float(train_cfg.get("lambda_loss", 100.0)) if variant.omega_loss else 0.0
    cfg = PhoneticPriorConfig(
        temperature=float(train_cfg.get("temperature", 0.2)),
        alpha=float(train_cfg.get("alpha_end", train_cfg.get("alpha", 3.5))),
        lambda_cov=float(train_cfg.get("lambda_cov", 10.0)),
        lambda_loss=lambda_loss,
        min_span=int(train_cfg.get("min_span", 4)),
        max_span=int(train_cfg.get("max_span", 10)),
        embedding_dim=int(train_cfg.get("embedding_dim", 100)),
        lr=float(train_cfg.get("learning_rate", 0.2)),
        p_o=float(train_cfg.get("p_o", 0.2)),
        seed=seed,
    )

    model = PhoneticPriorModel(chars_lost, chars_known, known_ipa_features=known_feats, config=cfg)
    _apply_mapping_prior(model, variant)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    num_steps = int(train_cfg.get("num_steps", 3000))
    anneal_steps = int(train_cfg.get("anneal_steps", 2000))
    alpha_start = float(train_cfg.get("alpha_start", 10.0))
    alpha_end = float(train_cfg.get("alpha_end", 3.5))
    log_interval = int(train_cfg.get("log_interval", 100))
    batch_size = int(train_cfg.get("batch_size", 8))

    # Pre-convert training text to list for random sampling.
    all_text = list(lost_training_text)
    rng = np.random.default_rng(seed)

    history: List[Dict[str, float]] = []
    for step in range(1, num_steps + 1):
        model.config.alpha = _annealed_alpha(alpha_start, alpha_end, anneal_steps, step)

        # Sample a mini-batch of inscriptions per step (Algorithm 1).
        # The paper processes a small sample per gradient update, not the
        # entire corpus.  This is critical for performance: WordBoundaryDP
        # is O(n × S × V) per inscription.
        if batch_size >= len(all_text):
            batch = all_text
        else:
            idxs = rng.choice(len(all_text), size=batch_size, replace=False)
            batch = [all_text[i] for i in idxs]

        out = train_one_step(model, optimizer, batch, known_vocab)
        if step == 1 or step == num_steps or step % log_interval == 0:
            history.append(
                {
                    "step": float(step),
                    "alpha": float(model.config.alpha),
                    "objective": out.objective,
                    "quality": out.quality,
                    "omega_cov": out.omega_cov,
                    "omega_loss": out.omega_loss,
                }
            )

    return TrainResult(model=model, history=history)


def _softmax_confidence(scores: Sequence[float], winner_idx: int) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    shifted = arr - float(np.max(arr))
    ex = np.exp(shifted)
    denom = float(np.sum(ex))
    if denom <= 0.0:
        return 0.0
    return float(ex[winner_idx] / denom)


def rank_queries(
    *,
    model: PhoneticPriorModel,
    queries: Sequence[str],
    gold_map: Mapping[str, Sequence[str]],
    known_vocab: Sequence[str],
    top_k: int,
) -> List[RankingRecord]:
    with torch.inference_mode():
        char_distr = model.compute_char_distr()
    records: List[RankingRecord] = []

    for idx, query in enumerate(queries, start=1):
        gold = [g for g in gold_map.get(query, []) if g]
        gold_set = set(gold)

        scored: List[Tuple[str, float, bool]] = []
        with torch.inference_mode():
            for known in known_vocab:
                score = float(model.edit_distance_dp(query, known, char_distr).item())
                scored.append((known, score, known in gold_set))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(1, top_k)]
        top_predictions = [x[0] for x in top]
        top_scores = [x[1] for x in top]

        first_positive_rank = None
        for rank, (_, _, is_pos) in enumerate(scored, start=1):
            if is_pos:
                first_positive_rank = rank
                break

        conf = _softmax_confidence([x[1] for x in scored], 0)

        records.append(
            RankingRecord(
                query_id=idx,
                query=query,
                gold_candidates=sorted(gold_set),
                top_predictions=top_predictions,
                top_scores=top_scores,
                first_positive_rank=first_positive_rank,
                top1_confidence=conf,
            )
        )

    return records


def compute_metrics(records: Sequence[RankingRecord], ks: Sequence[int]) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    n = len(records)
    metrics: Dict[str, float] = {
        "n_eval": float(n),
        "mrr": 0.0,
        "accuracy": 0.0,
    }
    if n == 0:
        for k in ks:
            metrics[f"p_at_{k}"] = 0.0
        return metrics

    mrr_total = 0.0
    correct_top1 = 0
    for rec in records:
        rank = rec.first_positive_rank
        if rank is not None:
            mrr_total += 1.0 / float(rank)
        if rank == 1:
            correct_top1 += 1

    metrics["mrr"] = mrr_total / float(n)
    metrics["accuracy"] = float(correct_top1) / float(n)
    for k in ks:
        hits = 0
        for rec in records:
            rank = rec.first_positive_rank
            if rank is not None and rank <= k:
                hits += 1
        metrics[f"p_at_{k}"] = float(hits) / float(n)
    return metrics


def build_curve(
    records: Sequence[RankingRecord],
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    n = len(records)
    if n == 0:
        return rows

    for t in sorted(float(x) for x in thresholds):
        covered = [r for r in records if r.top1_confidence >= t]
        coverage = float(len(covered)) / float(n)
        has_gold = any(bool(r.gold_candidates) for r in covered)
        if covered and has_gold:
            acc = float(sum(1 for r in covered if r.first_positive_rank == 1)) / float(len(covered))
        else:
            acc = float("nan")
        rows.append(
            {
                "threshold": t,
                "coverage": coverage,
                "accuracy": acc,
            }
        )
    return rows


def check_sanity(records: Sequence[RankingRecord], curve_rows: Optional[Sequence[Mapping[str, float]]] = None) -> None:
    metrics = compute_metrics(records, ks=[1, 10])
    if metrics.get("p_at_10", 0.0) + 1e-9 < metrics.get("p_at_1", 0.0):
        raise RuntimeError("Sanity check failed: P@10 < P@1.")

    if curve_rows:
        prev_cov: Optional[float] = None
        for row in curve_rows:
            cov = float(row["coverage"])
            if prev_cov is not None and cov > prev_cov + 1e-9:
                raise RuntimeError("Sanity check failed: coverage curve is not monotonic decreasing by threshold.")
            prev_cov = cov


def restart_seeds(seed_base: int, restarts: int) -> List[int]:
    return [int(seed_base) + i for i in range(int(restarts))]


def summarize_restarts(restart_rows: Sequence[Mapping[str, Any]], primary_metric: str) -> Dict[str, Any]:
    if not restart_rows:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "best": float("nan"),
            "best_seed": None,
        }

    values = [float(row[primary_metric]) for row in restart_rows]
    arr = np.asarray(values, dtype=np.float64)
    best_idx = int(np.argmax(arr))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "best": float(arr[best_idx]),
        "best_seed": int(restart_rows[best_idx]["seed"]),
    }


def ranking_records_to_rows(records: Sequence[RankingRecord], top_k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {
            "query_id": rec.query_id,
            "query": rec.query,
            "gold_candidates": "|".join(rec.gold_candidates),
            "first_positive_rank": rec.first_positive_rank,
            "top1_confidence": rec.top1_confidence,
        }
        for i in range(top_k):
            pred = rec.top_predictions[i] if i < len(rec.top_predictions) else ""
            score = rec.top_scores[i] if i < len(rec.top_scores) else ""
            row[f"pred_{i+1}"] = pred
            row[f"score_{i+1}"] = score
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fout:
        for row in rows:
            fout.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def closeness_score(curve_rows: Sequence[Mapping[str, float]], confidences: Sequence[float]) -> Dict[str, float]:
    if not curve_rows:
        return {
            "auc_coverage": 0.0,
            "auc_weighted_accuracy": float("nan"),
            "mean_confidence": 0.0,
            "closeness": 0.0,
        }

    xs = np.asarray([float(row["threshold"]) for row in curve_rows], dtype=np.float64)
    cov = np.asarray([float(row["coverage"]) for row in curve_rows], dtype=np.float64)
    acc = np.asarray([
        float(row["accuracy"]) if not math.isnan(float(row["accuracy"])) else 0.0 for row in curve_rows
    ], dtype=np.float64)

    auc_cov = float(np.trapz(cov, xs))
    auc_weighted = float(np.trapz(cov * acc, xs))
    mean_conf = float(np.mean(np.asarray(list(confidences), dtype=np.float64))) if confidences else 0.0
    closeness = auc_weighted if auc_weighted > 0.0 else auc_cov * mean_conf

    return {
        "auc_coverage": auc_cov,
        "auc_weighted_accuracy": auc_weighted,
        "mean_confidence": mean_conf,
        "closeness": closeness,
    }


def now_meta() -> Dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
    }
