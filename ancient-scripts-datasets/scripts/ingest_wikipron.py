#!/usr/bin/env python3
"""Ingest WikiPron pronunciation dictionary into per-language lexicon TSVs.

Scans sources/wikipron/data/scrape/tsv/ for all TSV files, extracts ISO 639-3
codes, prefers broad (phonemic) over narrow (phonetic) transcriptions, merges
multiple dialects per language, computes SCA sound classes, and writes to
data/training/lexicons/{iso}.tsv.

Dependencies: only Python standard library + cognate_pipeline.normalise.sound_class
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cognate_pipeline" / "src"))

from cognate_pipeline.normalise.sound_class import ipa_to_sound_class

WIKIPRON_TSV_DIR = ROOT / "sources" / "wikipron" / "data" / "scrape" / "tsv"
OUTPUT_DIR = ROOT / "data" / "training" / "lexicons"
METADATA_DIR = ROOT / "data" / "training" / "metadata"

# Filename pattern: {iso3}_{script}_{optional_dialect}_{broad|narrow}.tsv
# or {iso3}_{script}_{optional_dialect}_{broad|narrow}_filtered.tsv
FILENAME_RE = re.compile(
    r"^([a-z]{3})_([a-z]+)_(?:([a-z]+)_)?(broad|narrow)(?:_filtered)?\.tsv$"
)


def parse_filename(name: str) -> tuple[str, str, str, str] | None:
    """Parse WikiPron filename into (iso, script, dialect, type).

    Returns None if the filename doesn't match the expected pattern.
    """
    m = FILENAME_RE.match(name)
    if m:
        return m.group(1), m.group(2), m.group(3) or "", m.group(4)
    return None


def join_ipa_segments(segments: str) -> str:
    """Join space-separated IPA segments into a single IPA string."""
    return segments.replace(" ", "")


def normalize_ipa(ipa: str) -> str:
    """Basic IPA normalization: NFC, strip stress marks."""
    ipa = unicodedata.normalize("NFC", ipa)
    # Strip primary/secondary stress marks
    ipa = ipa.replace("\u02c8", "").replace("\u02cc", "")
    # Strip syllable breaks
    ipa = ipa.replace(".", "")
    return ipa.strip()


def inventory_files() -> dict[str, list[tuple[Path, str, str, str]]]:
    """Inventory all WikiPron TSV files, grouped by ISO code.

    Returns {iso: [(path, script, dialect, broad|narrow), ...]}
    """
    if not WIKIPRON_TSV_DIR.exists():
        print(f"ERROR: WikiPron TSV directory not found: {WIKIPRON_TSV_DIR}")
        return {}

    inventory: dict[str, list[tuple[Path, str, str, str]]] = defaultdict(list)
    skipped = 0

    for tsv_path in sorted(WIKIPRON_TSV_DIR.iterdir()):
        if not tsv_path.name.endswith(".tsv"):
            continue
        parsed = parse_filename(tsv_path.name)
        if parsed is None:
            skipped += 1
            continue
        iso, script, dialect, ttype = parsed
        # Skip filtered variants — use the unfiltered originals
        if "_filtered" in tsv_path.name:
            continue
        inventory[iso].append((tsv_path, script, dialect, ttype))

    print(f"  Inventoried {len(inventory)} languages, {skipped} files skipped")
    return dict(inventory)


def select_best_files(files: list[tuple[Path, str, str, str]]) -> list[Path]:
    """Select best files per language: prefer broad over narrow.

    If both broad and narrow exist, use only broad.
    If only narrow exists, use narrow.
    Merge across dialects and scripts.
    """
    broad_files = [f for f in files if f[3] == "broad"]
    narrow_files = [f for f in files if f[3] == "narrow"]

    if broad_files:
        return [f[0] for f in broad_files]
    return [f[0] for f in narrow_files]


def read_wikipron_tsv(path: Path) -> list[tuple[str, str]]:
    """Read a WikiPron TSV file. Returns list of (word, ipa) pairs."""
    entries = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                word = row[0].strip()
                ipa_raw = row[1].strip()
                if not word or not ipa_raw:
                    continue
                # Join space-separated segments
                ipa = join_ipa_segments(ipa_raw)
                ipa = normalize_ipa(ipa)
                if ipa:
                    entries.append((word, ipa))
    except Exception as e:
        print(f"  WARNING: Failed to read {path.name}: {e}")
    return entries


def ingest_all():
    """Main ingestion: read all WikiPron TSVs and write per-language lexicons."""
    print("=" * 80)
    print("WikiPron Ingestion")
    print("=" * 80)

    inventory = inventory_files()
    if not inventory:
        print("No WikiPron data found. Clone WikiPron first:")
        print("  git clone --depth 1 https://github.com/CUNY-CL/wikipron.git sources/wikipron")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    lang_stats = {}
    sca_unknown_total = 0
    sca_total_segments = 0

    for iso in sorted(inventory):
        files = inventory[iso]
        best = select_best_files(files)

        # Collect all entries for this language
        all_entries: list[tuple[str, str]] = []
        for fpath in best:
            entries = read_wikipron_tsv(fpath)
            all_entries.extend(entries)

        if not all_entries:
            continue

        # Deduplicate by (word, ipa)
        seen = set()
        unique_entries = []
        for word, ipa in all_entries:
            key = (word.lower(), ipa)
            if key not in seen:
                seen.add(key)
                unique_entries.append((word, ipa))

        # Compute SCA and write
        out_path = OUTPUT_DIR / f"{iso}.tsv"
        written = 0
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("Word\tIPA\tSCA\tSource\tConcept_ID\tCognate_Set_ID\n")
            for word, ipa in sorted(unique_entries, key=lambda x: x[0].lower()):
                sca = ipa_to_sound_class(ipa)
                # Track SCA quality
                unknown_count = sca.count("0")
                sca_total_segments += len(sca)
                sca_unknown_total += unknown_count
                f.write(f"{word}\t{ipa}\t{sca}\twikipron\t-\t-\n")
                written += 1

        lang_stats[iso] = written
        total_entries += written

        if written >= 1000:
            print(f"  {iso}: {written:>8,} entries")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Languages:     {len(lang_stats)}")
    print(f"  Total entries: {total_entries:,}")
    if sca_total_segments > 0:
        unknown_pct = 100 * sca_unknown_total / sca_total_segments
        print(f"  SCA unknown:   {sca_unknown_total:,} / {sca_total_segments:,} ({unknown_pct:.2f}%)")

    # Write source stats
    stats_path = METADATA_DIR / "wikipron_stats.tsv"
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        f.write("ISO\tEntries\n")
        for iso, count in sorted(lang_stats.items(), key=lambda x: -x[1]):
            f.write(f"{iso}\t{count}\n")

    # Top 20
    top = sorted(lang_stats.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 languages:")
    for iso, count in top:
        print(f"    {iso}: {count:>8,}")

    print(f"\n  Stats written to: {stats_path}")
    print("Done!")


if __name__ == "__main__":
    ingest_all()
