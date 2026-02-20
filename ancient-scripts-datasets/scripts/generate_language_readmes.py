#!/usr/bin/env python3
"""Generate per-language README profile markdown files.

Reads metadata, lexicons, and validation files, then writes one markdown
profile per language to data/training/language_profiles/{iso}.md.

Usage:
    python scripts/generate_language_readmes.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import langcodes

# Ensure cognate_pipeline package is importable (same pattern as build_validation_sets.py)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "cognate_pipeline" / "src"),
)

# Import constants from build_validation_sets
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from build_validation_sets import (
    ANCIENT,
    MEDIEVAL,
    RELIGIOUS_ALL,
    RELIGIOUS_SUBDOMAINS,
    classify_era,
    is_religious,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "data" / "training"
METADATA_DIR = TRAINING_DIR / "metadata"
LEXICONS_DIR = TRAINING_DIR / "lexicons"
VALIDATION_DIR = TRAINING_DIR / "validation"
OUTPUT_DIR = TRAINING_DIR / "language_profiles"

# Onomastic vs verbal classification for sub-domains
ONOMASTIC_SUBDOMAINS = {"core_religious", "sacred_places", "supernatural", "cosmic_spiritual"}
VERBAL_SUBDOMAINS = {"religious_verbs", "ritual_ceremony", "moral_ethical"}


# ---------------------------------------------------------------------------
# Phase 1: Load metadata
# ---------------------------------------------------------------------------

def load_languages_tsv() -> dict[str, dict[str, str]]:
    """Read languages.tsv → {iso: {family, glottocode, entries, sources}}."""
    path = METADATA_DIR / "languages.tsv"
    langs: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            iso = row["ISO"]
            langs[iso] = {
                "family": row.get("Family", ""),
                "glottocode": row.get("Glottocode", ""),
                "entries": row.get("Entries", "0"),
                "sources": row.get("Sources", ""),
            }
    return langs


def load_phylo_tree() -> dict:
    """Load phylo_tree.json."""
    path = VALIDATION_DIR / "phylo_tree.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_lang_paths(tree: dict) -> dict[str, list[str]]:
    """Map each ISO to its full path from root to leaf group."""
    paths: dict[str, list[str]] = {}

    def _walk(node, prefix: list[str]) -> None:
        if isinstance(node, list):
            for iso in node:
                paths[iso] = list(prefix)
        elif isinstance(node, dict):
            for key, child in node.items():
                _walk(child, prefix + [key])
        elif isinstance(node, str):
            paths[node] = list(prefix)

    _walk(tree, [])
    return paths


def get_display_name(iso: str) -> str:
    """Get human-readable language name via langcodes."""
    try:
        return langcodes.Language.get(iso).display_name()
    except Exception:
        return iso.upper()


# ---------------------------------------------------------------------------
# Phase 2: Index validation files
# ---------------------------------------------------------------------------

# All validation TSV files to index (relative to VALIDATION_DIR)
VALIDATION_FILES: list[str] = [
    "true_cognates_L1.tsv",
    "true_cognates_L2.tsv",
    "true_cognates_L3.tsv",
    "false_positives.tsv",
    "true_negatives.tsv",
    "borrowings.tsv",
    "timespan_ancient_ancient.tsv",
    "timespan_ancient_modern.tsv",
    "timespan_medieval_modern.tsv",
    "timespan_modern_modern.tsv",
]

PER_FAMILY_DIR = VALIDATION_DIR / "per_family"
RELIGIOUS_DIR = VALIDATION_DIR / "religious"

RELIGIOUS_FILES: list[str] = [
    "all_pairs.tsv",
    "true_cognates.tsv",
    "false_positives.tsv",
    "borrowings.tsv",
]


def index_validation_files() -> tuple[
    dict[str, Counter[str]],       # iso → {filename: count}
    dict[str, Counter[str]],       # iso → Counter(partner_iso)
    dict[str, dict[str, set[str]]],  # iso → {filename: set(concept_ids)}
    dict[str, dict[str, Counter[str]]],  # iso → {rel_filename: Counter(label)}
]:
    """Single-pass index of all validation TSVs.

    Returns:
        pair_counts: iso → {filename: pair_count}
        partners: iso → Counter(partner_iso)
        concept_sets: iso → {filename: set(concept_ids)}
        religious_labels: iso → {filename: Counter(label)}
    """
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    partners: dict[str, Counter[str]] = defaultdict(Counter)
    concept_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    religious_labels: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    def _index_file(filepath: Path, tag: str, is_religious_file: bool = False) -> None:
        if not filepath.exists():
            return
        with filepath.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                lang_a = row.get("Lang_A", "")
                lang_b = row.get("Lang_B", "")
                cid = row.get("Concept_ID", "")
                label = row.get("Label", "")

                if lang_a:
                    pair_counts[lang_a][tag] += 1
                    concept_sets[lang_a][tag].add(cid)
                    if lang_b:
                        partners[lang_a][lang_b] += 1
                    if is_religious_file:
                        religious_labels[lang_a][tag][label] += 1

                if lang_b:
                    pair_counts[lang_b][tag] += 1
                    concept_sets[lang_b][tag].add(cid)
                    if lang_a:
                        partners[lang_b][lang_a] += 1
                    if is_religious_file:
                        religious_labels[lang_b][tag][label] += 1

    # Core validation files
    for fname in VALIDATION_FILES:
        print(f"  Indexing {fname} ...")
        _index_file(VALIDATION_DIR / fname, fname)

    # Per-family files
    if PER_FAMILY_DIR.exists():
        for fp in sorted(PER_FAMILY_DIR.glob("*.tsv")):
            tag = f"per_family/{fp.name}"
            print(f"  Indexing {tag} ...")
            _index_file(fp, tag)

    # Religious core files
    for fname in RELIGIOUS_FILES:
        tag = f"religious/{fname}"
        print(f"  Indexing {tag} ...")
        _index_file(RELIGIOUS_DIR / fname, tag, is_religious_file=True)

    # Religious sub-domain files
    for sd_name in sorted(RELIGIOUS_SUBDOMAINS):
        fname = f"{sd_name}.tsv"
        tag = f"religious/{fname}"
        fp = RELIGIOUS_DIR / fname
        if fp.exists():
            print(f"  Indexing {tag} ...")
            _index_file(fp, tag, is_religious_file=True)

    return dict(pair_counts), dict(partners), dict(concept_sets), dict(religious_labels)


# ---------------------------------------------------------------------------
# Phase 3: Analyze lexicon per language
# ---------------------------------------------------------------------------

def analyze_lexicon(iso: str) -> dict:
    """Read a single lexicon TSV and return summary stats.

    Returns dict with keys: total_entries, unique_words, unique_concepts,
    religious_concepts (dict of subdomain → set of concept_ids),
    concept_list (set of all concept_ids).
    """
    path = LEXICONS_DIR / f"{iso}.tsv"
    if not path.exists():
        return {
            "total_entries": 0,
            "unique_words": 0,
            "unique_concepts": 0,
            "religious_concepts": {},
            "concept_list": set(),
        }

    words: set[str] = set()
    concepts: set[str] = set()
    total = 0

    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cid = row.get("Concept_ID", "").strip()
            if cid in ("", "-"):
                continue
            word = row.get("Word", "").strip()
            total += 1
            if word:
                words.add(word)
            concepts.add(cid)

    # Categorize concepts into religious sub-domains
    religious_concepts: dict[str, set[str]] = {}
    for sd_name, sd_set in RELIGIOUS_SUBDOMAINS.items():
        matching = set()
        for cid in concepts:
            if cid in sd_set or cid.upper() in {c.upper() for c in sd_set}:
                matching.add(cid)
        if matching:
            religious_concepts[sd_name] = matching

    return {
        "total_entries": total,
        "unique_words": len(words),
        "unique_concepts": len(concepts),
        "religious_concepts": religious_concepts,
        "concept_list": concepts,
    }


# ---------------------------------------------------------------------------
# Phase 4: Render markdown
# ---------------------------------------------------------------------------

def format_path(path_segments: list[str]) -> str:
    """Format a phylo path as 'Family > Branch > Sub-branch'."""
    if not path_segments:
        return "—"
    return " > ".join(s.replace("_", " ").title() for s in path_segments)


def render_profile(
    iso: str,
    meta: dict[str, str],
    lang_paths: dict[str, list[str]],
    pair_counts: dict[str, Counter[str]],
    partners_index: dict[str, Counter[str]],
    concept_sets: dict[str, dict[str, set[str]]],
    religious_labels: dict[str, dict[str, Counter[str]]],
    lexicon_stats: dict,
) -> str:
    """Render a single language profile as markdown."""
    display_name = get_display_name(iso)
    era = classify_era(iso)
    phylo_path = lang_paths.get(iso, [])
    family = meta.get("family", "unknown")
    glottocode = meta.get("glottocode", "")
    sources = meta.get("sources", "")

    counts = pair_counts.get(iso, Counter())
    my_partners = partners_index.get(iso, Counter())
    my_concepts = concept_sets.get(iso, {})
    my_rel_labels = religious_labels.get(iso, {})

    lines: list[str] = []

    # --- Header ---
    lines.append(f"# {display_name} (`{iso}`)")
    lines.append("")

    # --- 1. Overview table ---
    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| **ISO 639-3** | `{iso}` |")
    lines.append(f"| **Family** | {family.replace('_', ' ').title()} |")
    lines.append(f"| **Branch path** | {format_path(phylo_path)} |")
    lines.append(f"| **Glottocode** | {glottocode or '—'} |")
    lines.append(f"| **Era** | {era} |")
    lines.append(f"| **Sources** | {sources or '—'} |")
    lines.append("")

    # --- 2. Lexicon Summary ---
    lines.append("## Lexicon Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|------:|")
    lines.append(f"| Total entries | {lexicon_stats['total_entries']:,} |")
    lines.append(f"| Unique words | {lexicon_stats['unique_words']:,} |")
    lines.append(f"| Unique concepts | {lexicon_stats['unique_concepts']:,} |")
    lines.append("")

    # --- 3. Cognate Pair Participation ---
    lines.append("## Cognate Pair Participation")
    lines.append("")
    lines.append("| Validation Set | Pairs |")
    lines.append("|----------------|------:|")
    for fname, label in [
        ("true_cognates_L1.tsv", "True Cognates L1"),
        ("true_cognates_L2.tsv", "True Cognates L2"),
        ("true_cognates_L3.tsv", "True Cognates L3"),
        ("false_positives.tsv", "False Positives"),
        ("true_negatives.tsv", "True Negatives"),
        ("borrowings.tsv", "Borrowings"),
    ]:
        n = counts.get(fname, 0)
        lines.append(f"| {label} | {n:,} |")
    lines.append("")

    # --- 4. Timespan Distribution ---
    lines.append("## Timespan Distribution")
    lines.append("")
    lines.append("| Timespan | Pairs |")
    lines.append("|----------|------:|")
    for ts in ["ancient_ancient", "ancient_modern", "medieval_modern", "modern_modern"]:
        fname = f"timespan_{ts}.tsv"
        n = counts.get(fname, 0)
        lines.append(f"| {ts.replace('_', ' ').title()} | {n:,} |")
    lines.append("")

    # --- 5. Family-Internal Pairs ---
    family_files = sorted(
        k for k in counts if k.startswith("per_family/")
    )
    lines.append("## Family-Internal Pairs")
    lines.append("")
    if family_files:
        lines.append("| Family File | Pairs |")
        lines.append("|-------------|------:|")
        for ff in family_files:
            short = ff.replace("per_family/", "").replace(".tsv", "").replace("_", " ").title()
            n = counts[ff]
            lines.append(f"| {short} | {n:,} |")
    else:
        lines.append("This language does not appear in any per-family validation file.")
    lines.append("")

    # --- 6. Religious Domain ---
    rel_all = counts.get("religious/all_pairs.tsv", 0)
    lines.append("## Religious Domain")
    lines.append("")
    if rel_all > 0:
        lines.append(f"**Total religious pairs:** {rel_all:,}")
        lines.append("")

        # Breakdown by sub-domain
        lines.append("| Sub-domain | Category | Lexicon Concepts | Validation Pairs |")
        lines.append("|------------|----------|----------------:|-----------------:|")
        for sd_name in sorted(RELIGIOUS_SUBDOMAINS):
            category = "onomastic" if sd_name in ONOMASTIC_SUBDOMAINS else "verbal"
            lex_count = len(lexicon_stats["religious_concepts"].get(sd_name, set()))
            val_tag = f"religious/{sd_name}.tsv"
            val_count = counts.get(val_tag, 0)
            sd_display = sd_name.replace("_", " ").title()
            lines.append(f"| {sd_display} | {category} | {lex_count} | {val_count:,} |")
        lines.append("")

        # Label distribution within religious
        rel_label_agg: Counter[str] = Counter()
        for tag, label_counter in my_rel_labels.items():
            rel_label_agg.update(label_counter)
        if rel_label_agg:
            lines.append("**Religious label distribution:**")
            lines.append("")
            lines.append("| Label | Count |")
            lines.append("|-------|------:|")
            for label, cnt in rel_label_agg.most_common():
                lines.append(f"| {label} | {cnt:,} |")
            lines.append("")
    else:
        lines.append("This language does not appear in any religious validation file.")
        lines.append("")

    # --- 7. Top Partner Languages ---
    lines.append("## Top Partner Languages")
    lines.append("")
    if my_partners:
        top15 = my_partners.most_common(15)
        lines.append("| Partner | Pairs |")
        lines.append("|---------|------:|")
        for partner_iso, cnt in top15:
            partner_name = get_display_name(partner_iso)
            lines.append(f"| {partner_name} (`{partner_iso}`) | {cnt:,} |")
    else:
        lines.append("No partner languages found in validation data.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("generate_language_readmes.py — Per-Language Profile Generation")
    print("=" * 70)

    # Phase 1: Load metadata
    print("\nPhase 1: Loading metadata ...")
    languages = load_languages_tsv()
    print(f"  {len(languages)} languages in metadata")

    tree = load_phylo_tree()
    lang_paths = build_lang_paths(tree)
    print(f"  Phylo tree covers {len(lang_paths)} languages")

    # Phase 2: Index validation files
    print("\nPhase 2: Indexing validation files ...")
    pair_counts, partners_index, concept_sets, religious_labels = index_validation_files()
    print(f"  Indexed data for {len(pair_counts)} languages")

    # Phase 3 + 4: Analyze lexicon & render per language
    print("\nPhase 3+4: Analyzing lexicons and rendering profiles ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    isos = sorted(languages.keys())
    total = len(isos)
    for i, iso in enumerate(isos, 1):
        if i % 100 == 0 or i == total:
            print(f"  Processing {i}/{total} ({iso}) ...")

        lexicon_stats = analyze_lexicon(iso)
        meta = languages[iso]

        md = render_profile(
            iso, meta, lang_paths,
            pair_counts, partners_index, concept_sets,
            religious_labels, lexicon_stats,
        )

        out_path = OUTPUT_DIR / f"{iso}.md"
        out_path.write_text(md, encoding="utf-8")

    print(f"\n  Wrote {total} profiles to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
