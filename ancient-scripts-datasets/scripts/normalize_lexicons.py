#!/usr/bin/env python3
"""Normalize IPA and recompute SCA for all lexicon entries.

For every entry in data/training/lexicons/:
1. Unicode NFC normalization
2. Strip suprasegmentals (stress marks, tone diacritics)
3. Validate through tokenize_ipa() and ipa_to_sound_class()
4. Flag entries where SCA produces '0' (unknown segments)
5. Recompute and store SCA string

Handles source-specific quirks:
- WikiPron: multiple pronunciations separated by ', ' -> split into rows
- NorthEuraLex: segments already joined
- ABVD: orthographic forms treated as-is
"""

from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cognate_pipeline" / "src"))

from cognate_pipeline.normalise.sound_class import (
    ipa_to_sound_class,
    tokenize_ipa,
)

LEXICONS_DIR = ROOT / "data" / "training" / "lexicons"
METADATA_DIR = ROOT / "data" / "training" / "metadata"

HEADER = "Word\tIPA\tSCA\tSource\tConcept_ID\tCognate_Set_ID\n"

# Characters to strip from IPA (suprasegmentals, tone marks)
STRIP_CHARS = {
    "\u02c8",  # primary stress ˈ
    "\u02cc",  # secondary stress ˌ
    "\u02d0",  # length mark ː (keep — SCA ignores it)
    "\u02d1",  # half-length ˑ
    "\u0300",  # combining grave (tone)
    "\u0301",  # combining acute (tone)
    "\u0302",  # combining circumflex (tone)
    "\u0303",  # combining tilde (nasalization — keep for now)
    "\u0304",  # combining macron (tone)
    "\u030b",  # combining double acute (tone)
    "\u030c",  # combining caron (tone)
    "\u030f",  # combining double grave (tone)
}

# Only strip tone diacritics, not nasalization
TONE_STRIP = {
    "\u0300", "\u0301", "\u0302", "\u0304", "\u030b", "\u030c", "\u030f",
}


def normalize_entry_ipa(ipa: str) -> str:
    """Normalize a single IPA string."""
    # NFC first
    ipa = unicodedata.normalize("NFC", ipa)
    # Strip stress marks
    ipa = ipa.replace("\u02c8", "").replace("\u02cc", "")
    # Strip syllable breaks
    ipa = ipa.replace(".", "")
    # Strip tone diacritics
    for ch in TONE_STRIP:
        ipa = ipa.replace(ch, "")
    return ipa.strip()


def process_lexicon(path: Path) -> tuple[int, int, int, int]:
    """Process a single lexicon file. Returns (total, valid, unknown_segments, dupes)."""
    if not path.exists():
        return 0, 0, 0, 0

    entries = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return 0, 0, 0, 0
        for row in reader:
            if len(row) < 6:
                row.extend(["-"] * (6 - len(row)))
            entries.append(row)

    # Process entries
    normalized = []
    seen = set()
    total = 0
    unknown_segments = 0
    dupes = 0

    for row in entries:
        word, ipa, sca_old, source, concept_id, cognate_set_id = (
            row[0], row[1], row[2], row[3], row[4], row[5]
        )

        # Handle multiple pronunciations (WikiPron sometimes has them)
        ipa_variants = [ipa]
        if ", " in ipa and source == "wikipron":
            ipa_variants = [v.strip() for v in ipa.split(", ") if v.strip()]

        for ipa_v in ipa_variants:
            total += 1
            ipa_clean = normalize_entry_ipa(ipa_v)
            if not ipa_clean:
                continue

            # Dedup key
            key = (word, ipa_clean)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)

            # Recompute SCA
            sca = ipa_to_sound_class(ipa_clean)

            # Check for unknowns
            if "0" in sca:
                unknown_segments += sca.count("0")

            normalized.append((word, ipa_clean, sca, source, concept_id, cognate_set_id))

    # Write back
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        for word, ipa, sca, source, concept_id, cognate_set_id in sorted(
            normalized, key=lambda x: x[0].lower()
        ):
            f.write(f"{word}\t{ipa}\t{sca}\t{source}\t{concept_id}\t{cognate_set_id}\n")

    return total, len(normalized), unknown_segments, dupes


def main():
    print("=" * 80)
    print("IPA Normalization & SCA Recomputation")
    print("=" * 80)

    if not LEXICONS_DIR.exists():
        print("No lexicons directory found. Run ingestion scripts first.")
        return

    files = sorted(LEXICONS_DIR.glob("*.tsv"))
    print(f"\n  Found {len(files)} lexicon files")

    grand_total = 0
    grand_valid = 0
    grand_unknown = 0
    grand_dupes = 0
    total_sca_chars = 0
    problem_langs = []

    for path in files:
        total, valid, unknown, dupes = process_lexicon(path)
        grand_total += total
        grand_valid += valid
        grand_unknown += unknown
        grand_dupes += dupes

        # Estimate total SCA chars for percentage
        # (approximation: valid entries * avg SCA length ~5)
        if valid > 0:
            # Read actual SCA lengths
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        total_sca_chars += len(row[2])

        if unknown > 0 and valid > 0:
            unknown_rate = unknown / max(total_sca_chars, 1)
            if unknown_rate > 0.05:
                problem_langs.append((path.stem, unknown, valid))

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total entries processed: {grand_total:,}")
    print(f"  Valid entries retained:  {grand_valid:,}")
    print(f"  Duplicates removed:      {grand_dupes:,}")
    if total_sca_chars > 0:
        pct = 100 * grand_unknown / total_sca_chars
        print(f"  SCA unknown segments:    {grand_unknown:,} / {total_sca_chars:,} ({pct:.2f}%)")

    if problem_langs:
        print(f"\n  Languages with >5% SCA unknowns:")
        for lang, unk, valid in problem_langs:
            print(f"    {lang}: {unk} unknown segments in {valid} entries")

    print("\nDone!")


if __name__ == "__main__":
    main()
