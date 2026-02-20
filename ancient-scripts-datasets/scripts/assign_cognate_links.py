#!/usr/bin/env python3
"""Assign cognate links from expert annotations and automated scoring.

1. Expert cognates from ABVD (Cognacy) and sinotibetan (COGID)
2. Concept-aligned pairs within families scored by Levenshtein
3. WOLD borrowing pairs

Outputs:
  data/training/cognate_pairs/cognate_pairs_inherited.tsv
  data/training/cognate_pairs/cognate_pairs_similarity.tsv
  data/training/cognate_pairs/cognate_pairs_borrowing.tsv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cognate_pipeline" / "src"))

from cognate_pipeline.cognate.baseline_levenshtein import normalised_similarity
from cognate_pipeline.normalise.sound_class import ipa_to_sound_class

LEXICONS_DIR = ROOT / "data" / "training" / "lexicons"
COGNATE_DIR = ROOT / "data" / "training" / "cognate_pairs"
SOURCES_DIR = ROOT / "sources"

FAMILY_MAP_PATH = ROOT / "cognate_pipeline" / "src" / "cognate_pipeline" / "cognate" / "family_map.json"

PAIR_HEADER = "Lang_A\tWord_A\tIPA_A\tLang_B\tWord_B\tIPA_B\tConcept_ID\tRelationship\tScore\tSource\n"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_lexicon(path: Path) -> list[dict]:
    """Read a lexicon TSV file."""
    entries = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries.append(row)
    return entries


def extract_expert_cognates() -> list[tuple]:
    """Extract expert cognate pairs from lexicon files (ABVD + sinotibetan)."""
    pairs = []

    # Group entries by cognate set ID across all lexicons
    cognate_sets: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    # (iso, word, ipa, sca, concept_id)

    for path in sorted(LEXICONS_DIR.glob("*.tsv")):
        iso = path.stem
        for entry in read_lexicon(path):
            cog_id = entry.get("Cognate_Set_ID", "-")
            if cog_id == "-" or not cog_id:
                continue
            source = entry.get("Source", "")
            if source not in ("abvd", "sinotibetan"):
                continue
            cognate_sets[cog_id].append((
                iso,
                entry.get("Word", ""),
                entry.get("IPA", ""),
                entry.get("SCA", ""),
                entry.get("Concept_ID", "-"),
            ))

    # Generate pairs within each cognate set
    for cog_id, members in cognate_sets.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = members[i]
                b = members[j]
                if a[0] == b[0]:  # skip same-language pairs
                    continue
                score = normalised_similarity(a[3], b[3])
                source = "abvd" if cog_id.startswith("abvd_") else "sinotibetan"
                pairs.append((
                    a[0], a[1], a[2],  # lang_a, word_a, ipa_a
                    b[0], b[1], b[2],  # lang_b, word_b, ipa_b
                    a[4],              # concept_id
                    "expert_cognate",
                    round(score, 4),
                    source,
                ))

    return pairs


def extract_concept_pairs(family_map: dict) -> tuple[list[tuple], list[tuple]]:
    """Extract concept-aligned pairs within families.

    Returns (inherited_pairs, similarity_pairs).
    """
    # Load all concept-annotated entries grouped by (family, concept)
    family_concept: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    # (iso, word, ipa, sca)

    for path in sorted(LEXICONS_DIR.glob("*.tsv")):
        iso = path.stem
        family = family_map.get(iso, "unknown")
        if family == "unknown":
            continue
        for entry in read_lexicon(path):
            concept = entry.get("Concept_ID", "-")
            if concept == "-" or not concept:
                continue
            sca = entry.get("SCA", "")
            if not sca:
                continue
            family_concept[(family, concept)].append((
                iso,
                entry.get("Word", ""),
                entry.get("IPA", ""),
                sca,
            ))

    inherited = []
    similarity = []

    for (family, concept), members in family_concept.items():
        if len(members) < 2:
            continue

        # Limit pairs per concept to avoid O(n^2) explosion
        # For large sets, sample representatives
        members_sample = members[:50]  # cap at 50 entries per concept

        for i in range(len(members_sample)):
            for j in range(i + 1, len(members_sample)):
                a = members_sample[i]
                b = members_sample[j]
                if a[0] == b[0]:  # skip same-language
                    continue

                score = normalised_similarity(a[3], b[3])

                if score >= 0.5:
                    inherited.append((
                        a[0], a[1], a[2],
                        b[0], b[1], b[2],
                        concept,
                        "cognate_inherited",
                        round(score, 4),
                        f"concept_align_{family}",
                    ))
                elif score >= 0.3:
                    similarity.append((
                        a[0], a[1], a[2],
                        b[0], b[1], b[2],
                        concept,
                        "similarity_only",
                        round(score, 4),
                        f"concept_align_{family}",
                    ))

    return inherited, similarity


def extract_wold_borrowings() -> list[tuple]:
    """Extract borrowing relationships from WOLD."""
    cldf_dir = SOURCES_DIR / "wold" / "cldf"
    if not cldf_dir.exists():
        return []

    # Use forms.csv Borrowed column to identify borrowed items
    return _extract_borrowings_from_forms(cldf_dir)


def _extract_borrowings_from_forms(cldf_dir: Path) -> list[tuple]:
    """Extract borrowing indicators from WOLD forms.csv."""
    forms_path = cldf_dir / "forms.csv"
    if not forms_path.exists():
        return []

    # Build language map
    wold_lang_map = {}
    for row in _read_csv(cldf_dir / "languages.csv"):
        iso = row.get("ISO639P3code", "")
        if iso:
            wold_lang_map[row["ID"]] = iso

    # Build parameter map
    param_map = {}
    for row in _read_csv(cldf_dir / "parameters.csv"):
        gloss = row.get("Concepticon_Gloss", row.get("Name", row["ID"]))
        param_map[row["ID"]] = gloss

    # Collect forms with borrowing annotations
    # Group by concept for cross-language pairing
    concept_forms: dict[str, list[tuple]] = defaultdict(list)

    for row in _read_csv(forms_path):
        lang_id = row.get("Language_ID", "")
        iso = wold_lang_map.get(lang_id)
        if not iso:
            continue

        borrowed = row.get("Borrowed", "").strip()
        if not borrowed or "no evidence" in borrowed or "very little" in borrowed:
            continue

        segments = row.get("Segments", "")
        if not segments:
            continue

        ipa = "".join(p for p in segments.split() if p not in ("^", "$", "+", "#", "_"))
        if not ipa:
            continue

        param_id = row.get("Parameter_ID", "")
        concept = param_map.get(param_id, param_id)
        word = row.get("Form", "")
        sca = ipa_to_sound_class(ipa)

        concept_forms[concept].append((iso, word, ipa, sca, borrowed))

    # Generate borrowing pairs (forms from different languages sharing a concept)
    pairs = []
    for concept, forms in concept_forms.items():
        if len(forms) < 2:
            continue
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                a = forms[i]
                b = forms[j]
                if a[0] == b[0]:
                    continue
                score = normalised_similarity(a[3], b[3])
                if score >= 0.3:
                    pairs.append((
                        a[0], a[1], a[2],
                        b[0], b[1], b[2],
                        concept,
                        "borrowing",
                        round(score, 4),
                        "wold",
                    ))

    return pairs


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_pairs(path: Path, pairs: list[tuple]):
    """Write cognate pairs to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(PAIR_HEADER)
        for pair in pairs:
            f.write("\t".join(str(x) for x in pair) + "\n")


def main():
    print("=" * 80)
    print("Cognate Link Assignment")
    print("=" * 80)

    family_map = load_json(FAMILY_MAP_PATH)

    # Phase 1: Expert cognates
    print("\n  [Expert Cognates]")
    expert_pairs = extract_expert_cognates()
    print(f"  Expert cognate pairs: {len(expert_pairs):,}")

    # Phase 2: Concept-aligned pairs
    print("\n  [Concept-Aligned Pairs]")
    inherited, similarity = extract_concept_pairs(family_map)
    print(f"  Inherited pairs:  {len(inherited):,}")
    print(f"  Similarity pairs: {len(similarity):,}")

    # Phase 3: WOLD borrowings
    print("\n  [WOLD Borrowings]")
    borrowing_pairs = extract_wold_borrowings()
    print(f"  Borrowing pairs: {len(borrowing_pairs):,}")

    # Merge expert into inherited
    all_inherited = expert_pairs + inherited

    # Write output files
    COGNATE_DIR.mkdir(parents=True, exist_ok=True)
    write_pairs(COGNATE_DIR / "cognate_pairs_inherited.tsv", all_inherited)
    write_pairs(COGNATE_DIR / "cognate_pairs_similarity.tsv", similarity)
    write_pairs(COGNATE_DIR / "cognate_pairs_borrowing.tsv", borrowing_pairs)

    total_pairs = len(all_inherited) + len(similarity) + len(borrowing_pairs)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total cognate pairs: {total_pairs:,}")
    print(f"    Inherited:   {len(all_inherited):,}")
    print(f"    Similarity:  {len(similarity):,}")
    print(f"    Borrowing:   {len(borrowing_pairs):,}")
    print(f"\n  Output: {COGNATE_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
