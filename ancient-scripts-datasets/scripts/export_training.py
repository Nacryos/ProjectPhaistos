#!/usr/bin/env python3
"""Export training data for PhoneticPriorModel from lexicon TSVs and validation data.

Creates model-compatible data files for the decipherment model:
  - lost.txt: One lost-language word per line
  - known_<lang>.txt: Full known-language vocabulary
  - ground_truth.cog: Tab-separated cognate pairs for evaluation
  - negatives.tsv: Non-cognate negative pairs for evaluation

Usage:
  # Export from validation data (best for ancient/historical languages)
  python scripts/export_training.py --branch germanic --out exports/germanic/

  # Export a specific pair using lexicon concept matching
  python scripts/export_training.py --lost fin --known hun --out exports/fin_hun/

  # With negative samples (3x ratio)
  python scripts/export_training.py --branch celtic --negatives 3 --out exports/celtic/

Dependencies: only Python standard library
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
LEXICONS = ROOT / "data" / "training" / "lexicons"
VALIDATION = ROOT / "data" / "validation"
METADATA = ROOT / "data" / "training" / "metadata"


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def read_lexicon(iso: str) -> List[Dict[str, str]]:
    """Read a per-language lexicon TSV."""
    path = LEXICONS / f"{iso}.tsv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_validation_tsv(branch: str) -> List[Dict[str, str]]:
    """Read a validation TSV file (Language_ID, Parameter_ID, Form, IPA, Glottocode)."""
    path = VALIDATION / f"{branch}.tsv"
    if not path.exists():
        print(f"  ERROR: Validation file not found: {path}", file=sys.stderr)
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _index_by_key(rows: List[Dict[str, str]], key: str, exclude: str = "-") -> Dict[str, List[Dict[str, str]]]:
    idx: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        v = row.get(key, exclude)
        if v and v != exclude:
            idx[v].append(row)
    return idx


def match_lexicon_pairs(
    lost_rows: List[Dict[str, str]],
    known_rows: List[Dict[str, str]],
) -> Tuple[List[Tuple[str, str, str]], str]:
    """Match lost-known pairs from lexicon TSVs.

    Tries Cognate_Set_ID first, then Concept_ID.
    Returns (pairs, method) where pairs is [(lost_ipa, known_ipa, match_id)].
    """
    # Try cognate set matching
    lost_cog = _index_by_key(lost_rows, "Cognate_Set_ID")
    known_cog = _index_by_key(known_rows, "Cognate_Set_ID")
    shared = set(lost_cog) & set(known_cog)

    if shared:
        pairs = []
        for cid in sorted(shared):
            for lr in lost_cog[cid]:
                lip = lr.get("IPA", "").strip()
                if not lip:
                    continue
                for kr in known_cog[cid]:
                    kip = kr.get("IPA", "").strip()
                    if kip:
                        pairs.append((lip, kip, cid))
        if pairs:
            return sorted(set(pairs)), "cognate_set"

    # Fall back to concept matching
    lost_con = _index_by_key(lost_rows, "Concept_ID")
    known_con = _index_by_key(known_rows, "Concept_ID")
    shared = set(lost_con) & set(known_con)

    pairs = []
    for cid in sorted(shared):
        for lr in lost_con[cid]:
            lip = lr.get("IPA", "").strip()
            if not lip:
                continue
            for kr in known_con[cid]:
                kip = kr.get("IPA", "").strip()
                if kip:
                    pairs.append((lip, kip, cid))
    return sorted(set(pairs)), "concept"


def match_validation_pairs(
    rows: List[Dict[str, str]],
    lost_iso: str,
    known_iso: str,
) -> List[Tuple[str, str, str]]:
    """Match pairs from a validation TSV by shared Parameter_ID (concept)."""
    by_lang: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        lang = row.get("Language_ID", "").strip()
        concept = row.get("Parameter_ID", "").strip()
        ipa = row.get("IPA", "").strip()
        if lang and concept and ipa:
            by_lang[lang][concept].append(ipa)

    lost_concepts = by_lang.get(lost_iso, {})
    known_concepts = by_lang.get(known_iso, {})

    shared = set(lost_concepts) & set(known_concepts)
    pairs = []
    for cid in sorted(shared):
        for lip in lost_concepts[cid]:
            for kip in known_concepts[cid]:
                pairs.append((lip, kip, cid))
    return sorted(set(pairs))


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def generate_negatives(
    pairs: List[Tuple[str, str, str]],
    known_vocab: List[str],
    ratio: int = 3,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Generate negative (non-cognate) pairs by random known-word substitution."""
    rng = random.Random(seed)
    negatives = []
    for lost_ipa, known_ipa, _ in pairs:
        candidates = [v for v in known_vocab if v != known_ipa]
        if not candidates:
            continue
        n = min(ratio, len(candidates))
        for neg_known in rng.sample(candidates, n):
            negatives.append((lost_ipa, neg_known))
    return negatives


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def hash_split(key: str, train: float = 0.8, val: float = 0.1) -> str:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < int(train * 100):
        return "train"
    elif h < int((train + val) * 100):
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _write_export(
    pair_dir: Path,
    lost_iso: str,
    known_iso: str,
    pairs: List[Tuple[str, str, str]],
    lost_vocab: List[str],
    known_vocab: List[str],
    match_method: str,
    neg_ratio: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Write all export files for a single language pair."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    # lost.txt
    (pair_dir / "lost.txt").write_text("\n".join(lost_vocab) + "\n", encoding="utf-8")

    # known_<iso>.txt
    (pair_dir / f"known_{known_iso}.txt").write_text("\n".join(known_vocab) + "\n", encoding="utf-8")

    # ground_truth.cog
    gold_map: Dict[str, Set[str]] = defaultdict(set)
    for lip, kip, _ in pairs:
        gold_map[lip].add(kip)

    matched_known: Set[str] = set()
    with open(pair_dir / "ground_truth.cog", "w", encoding="utf-8") as f:
        f.write(f"{lost_iso}\t{known_iso}\n")
        for lip in sorted(gold_map):
            kforms = sorted(gold_map[lip])
            f.write(f"{lip}\t{'|'.join(kforms)}\n")
            matched_known.update(kforms)
        for k in known_vocab:
            if k not in matched_known:
                f.write(f"_\t{k}\n")

    # splits.json
    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, lip in enumerate(sorted(gold_map)):
        splits[hash_split(lip)].append(i)
    (pair_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    # negatives.tsv
    neg_count = 0
    if neg_ratio > 0:
        negs = generate_negatives(pairs, known_vocab, ratio=neg_ratio, seed=seed)
        with open(pair_dir / "negatives.tsv", "w", encoding="utf-8") as f:
            f.write(f"{lost_iso}\t{known_iso}\tlabel\n")
            for lip, kip in negs:
                f.write(f"{lip}\t{kip}\t0\n")
        neg_count = len(negs)

    meta: Dict[str, Any] = {
        "lost_iso": lost_iso,
        "known_iso": known_iso,
        "match_method": match_method,
        "num_pairs": len(pairs),
        "num_unique_lost": len(gold_map),
        "num_lost_vocab": len(lost_vocab),
        "num_known_vocab": len(known_vocab),
        "num_negatives": neg_count,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    (pair_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"    Method: {match_method} | Pairs: {len(pairs)} ({len(gold_map)} unique lost)")
    print(f"    Lost vocab: {len(lost_vocab)}, Known vocab: {len(known_vocab)}")
    print(f"    Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    if neg_count:
        print(f"    Negatives: {neg_count}")

    return meta


def export_lexicon_pair(
    lost_iso: str,
    known_iso: str,
    out_dir: Path,
    neg_ratio: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Export a language pair using lexicon TSV concept/cognate matching."""
    print(f"\n  Exporting {lost_iso} -> {known_iso} (lexicon)")

    lost_rows = read_lexicon(lost_iso)
    known_rows = read_lexicon(known_iso)
    if not lost_rows or not known_rows:
        print(f"  SKIP: Missing lexicon data for {lost_iso} or {known_iso}")
        return {}

    pairs, method = match_lexicon_pairs(lost_rows, known_rows)
    if not pairs:
        print(f"  SKIP: No matching pairs for {lost_iso}-{known_iso}")
        return {}

    lost_vocab = sorted({r.get("IPA", "").strip() for r in lost_rows if r.get("IPA", "").strip()})
    known_vocab = sorted({r.get("IPA", "").strip() for r in known_rows if r.get("IPA", "").strip()})

    return _write_export(
        out_dir / f"{lost_iso}_{known_iso}",
        lost_iso, known_iso, pairs, lost_vocab, known_vocab,
        method, neg_ratio, seed,
    )


def export_validation_pair(
    val_rows: List[Dict[str, str]],
    lost_iso: str,
    known_iso: str,
    out_dir: Path,
    neg_ratio: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Export a language pair from validation TSV data."""
    print(f"\n  Exporting {lost_iso} -> {known_iso} (validation)")

    pairs = match_validation_pairs(val_rows, lost_iso, known_iso)
    if not pairs:
        print(f"  SKIP: No matching pairs for {lost_iso}-{known_iso}")
        return {}

    # Build full vocab from validation + lexicon (lexicon provides full vocabulary)
    lost_vocab_set: Set[str] = set()
    known_vocab_set: Set[str] = set()

    # From validation
    for row in val_rows:
        lang = row.get("Language_ID", "").strip()
        ipa = row.get("IPA", "").strip()
        if ipa:
            if lang == lost_iso:
                lost_vocab_set.add(ipa)
            elif lang == known_iso:
                known_vocab_set.add(ipa)

    # Supplement from lexicons if available
    for row in read_lexicon(lost_iso):
        ipa = row.get("IPA", "").strip()
        if ipa:
            lost_vocab_set.add(ipa)
    for row in read_lexicon(known_iso):
        ipa = row.get("IPA", "").strip()
        if ipa:
            known_vocab_set.add(ipa)

    lost_vocab = sorted(lost_vocab_set)
    known_vocab = sorted(known_vocab_set)

    return _write_export(
        out_dir / f"{lost_iso}_{known_iso}",
        lost_iso, known_iso, pairs, lost_vocab, known_vocab,
        "validation", neg_ratio, seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export training data for PhoneticPriorModel")
    parser.add_argument("--lost", help="Lost language ISO code")
    parser.add_argument("--known", help="Known language ISO code")
    parser.add_argument("--branch", help="Validation branch (germanic, celtic, etc.)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--negatives", type=int, default=0, help="Negative sample ratio (0=none)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.branch:
        val_rows = read_validation_tsv(args.branch)
        if not val_rows:
            sys.exit(1)

        langs = sorted({r["Language_ID"].strip() for r in val_rows if r.get("Language_ID", "").strip()})
        print(f"Branch {args.branch}: {len(langs)} languages: {langs}")
        all_meta = []

        for i, lost in enumerate(langs):
            for known in langs[i + 1:]:
                # Both directions
                for l, k in [(lost, known), (known, lost)]:
                    meta = export_validation_pair(val_rows, l, k, out_dir, args.negatives, args.seed)
                    if meta:
                        all_meta.append(meta)

        summary = {
            "branch": args.branch,
            "languages": langs,
            "num_pairs_exported": len(all_meta),
            "total_cognate_pairs": sum(m.get("num_pairs", 0) for m in all_meta),
            "pairs": all_meta,
        }
        (out_dir / "branch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"Branch {args.branch}: {len(all_meta)} direction-pairs exported")
        print(f"Total cognate pairs: {summary['total_cognate_pairs']:,}")

    elif args.lost and args.known:
        meta = export_lexicon_pair(args.lost, args.known, out_dir, args.negatives, args.seed)
        if not meta:
            print("No data exported.")
            sys.exit(1)
        print(f"\nExported to: {out_dir / f'{args.lost}_{args.known}'}")
    else:
        parser.error("Provide either --lost/--known or --branch")


if __name__ == "__main__":
    main()
