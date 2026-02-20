#!/usr/bin/env python3
"""Assemble final per-language lexicons and generate metadata.

For each language (by ISO code):
1. Read lexicon file from data/training/lexicons/{iso}.tsv
2. Verify deduplication by (Word, IPA)
3. Sort by Word alphabetically
4. Generate data/training/metadata/languages.tsv
5. Generate data/training/metadata/source_stats.tsv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cognate_pipeline" / "src"))

LEXICONS_DIR = ROOT / "data" / "training" / "lexicons"
METADATA_DIR = ROOT / "data" / "training" / "metadata"

# Load family_map for family assignments
FAMILY_MAP_PATH = ROOT / "cognate_pipeline" / "src" / "cognate_pipeline" / "cognate" / "family_map.json"
LANGUAGE_MAP_PATH = ROOT / "cognate_pipeline" / "src" / "cognate_pipeline" / "ingest" / "language_map.json"

HEADER = "Word\tIPA\tSCA\tSource\tConcept_ID\tCognate_Set_ID\n"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def assemble_lexicon(path: Path) -> tuple[int, int, dict[str, int]]:
    """Re-sort and final dedup a lexicon. Returns (total_in, total_out, source_counts)."""
    entries = []
    seen = set()
    dupes = 0
    source_counts: dict[str, int] = defaultdict(int)

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 6:
                row.extend(["-"] * (6 - len(row)))
            word, ipa, sca, source, concept_id, cognate_set_id = (
                row[0], row[1], row[2], row[3], row[4], row[5]
            )
            key = (word, ipa)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            entries.append((word, ipa, sca, source, concept_id, cognate_set_id))
            source_counts[source] += 1

    total_in = len(entries) + dupes

    # Sort alphabetically by word (case-insensitive)
    entries.sort(key=lambda x: x[0].lower())

    # Write back
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        for word, ipa, sca, source, concept_id, cognate_set_id in entries:
            f.write(f"{word}\t{ipa}\t{sca}\t{source}\t{concept_id}\t{cognate_set_id}\n")

    return total_in, len(entries), dict(source_counts)


def main():
    print("=" * 80)
    print("Lexicon Assembly & Metadata Generation")
    print("=" * 80)

    if not LEXICONS_DIR.exists():
        print("No lexicons directory found.")
        return

    family_map = load_json(FAMILY_MAP_PATH)
    language_map = load_json(LANGUAGE_MAP_PATH)

    files = sorted(LEXICONS_DIR.glob("*.tsv"))
    print(f"\n  Found {len(files)} lexicon files")

    lang_metadata = []
    all_source_stats = []
    grand_total = 0
    grand_dupes = 0

    for path in files:
        iso = path.stem
        total_in, total_out, source_counts = assemble_lexicon(path)
        dupes = total_in - total_out
        grand_total += total_out
        grand_dupes += dupes

        family = family_map.get(iso, "unknown")
        glottocode = language_map.get(iso, "")

        lang_metadata.append({
            "iso": iso,
            "family": family,
            "glottocode": glottocode,
            "entries": total_out,
            "sources": ",".join(sorted(source_counts.keys())),
        })

        for source, count in source_counts.items():
            all_source_stats.append({
                "iso": iso,
                "source": source,
                "entries": count,
            })

    # Write languages.tsv
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    lang_path = METADATA_DIR / "languages.tsv"
    with open(lang_path, "w", encoding="utf-8", newline="") as f:
        f.write("ISO\tFamily\tGlottocode\tEntries\tSources\n")
        for m in sorted(lang_metadata, key=lambda x: -x["entries"]):
            f.write(f"{m['iso']}\t{m['family']}\t{m['glottocode']}\t{m['entries']}\t{m['sources']}\n")

    # Write source_stats.tsv
    stats_path = METADATA_DIR / "source_stats.tsv"
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        f.write("ISO\tSource\tEntries\n")
        for s in sorted(all_source_stats, key=lambda x: (x["iso"], x["source"])):
            f.write(f"{s['iso']}\t{s['source']}\t{s['entries']}\n")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Languages:        {len(lang_metadata)}")
    print(f"  Total entries:    {grand_total:,}")
    print(f"  Duplicates:       {grand_dupes:,}")

    # Top 20 by size
    top = sorted(lang_metadata, key=lambda x: -x["entries"])[:20]
    print(f"\n  Top 20 languages:")
    for m in top:
        print(f"    {m['iso']} ({m['family']}): {m['entries']:>8,} entries [{m['sources']}]")

    # Family distribution
    fam_counts: dict[str, int] = defaultdict(int)
    for m in lang_metadata:
        fam_counts[m["family"]] += m["entries"]
    print(f"\n  Entries by family:")
    for fam, count in sorted(fam_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {fam}: {count:>8,}")

    print(f"\n  Metadata: {lang_path}")
    print(f"  Stats:    {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
