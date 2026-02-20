#!/usr/bin/env python3
"""Extract ALL data from CLDF sources — no concept filtering.

Reads NorthEuraLex (all 1,016 concepts), WOLD (all 1,814 concepts),
ABVD (all 210 concepts, top Austronesian languages), and sinotibetan
(all concepts). Writes to data/training/lexicons/{iso}.tsv files.

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

SOURCES = ROOT / "sources"
OUTPUT_DIR = ROOT / "data" / "training" / "lexicons"
METADATA_DIR = ROOT / "data" / "training" / "metadata"

# Header for lexicon files
HEADER = "Word\tIPA\tSCA\tSource\tConcept_ID\tCognate_Set_ID\n"


def read_cldf_csv(path: Path) -> list[dict[str, str]]:
    """Read a CLDF CSV file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def segments_to_ipa(segments: str) -> str:
    """Convert CLDF Segments column (space-separated) to IPA string."""
    if not segments or not segments.strip():
        return ""
    parts = segments.split()
    cleaned = [p for p in parts if p not in ("^", "$", "+", "#", "_")]
    return "".join(cleaned)


def normalize_ipa(ipa: str) -> str:
    """Basic IPA normalization: NFC, strip stress marks."""
    ipa = unicodedata.normalize("NFC", ipa)
    ipa = ipa.replace("\u02c8", "").replace("\u02cc", "")
    ipa = ipa.replace(".", "")
    return ipa.strip()


def form_to_pseudo_ipa(form: str) -> str:
    """For repos without IPA, use orthographic form as pseudo-IPA."""
    if not form:
        return ""
    form = form.lower().strip()
    form = re.sub(r"\(.*?\)", "", form)
    if "," in form:
        form = form.split(",")[0].strip()
    if "/" in form:
        form = form.split("/")[0].strip()
    form = re.sub(r"[^a-zA-Z\u0250-\u02AF\u0300-\u036F\u0361]", "", form)
    return form


def append_to_lexicon(
    lang_entries: dict[str, list[tuple[str, str, str, str, str, str]]],
    iso: str,
    word: str,
    ipa: str,
    source: str,
    concept_id: str = "-",
    cognate_set_id: str = "-",
):
    """Append an entry to the in-memory language buffer."""
    if not word or not ipa:
        return
    ipa = normalize_ipa(ipa)
    if not ipa:
        return
    sca = ipa_to_sound_class(ipa)
    lang_entries[iso].append((word, ipa, sca, source, concept_id, cognate_set_id))


def extract_northeuralex(lang_entries: dict) -> int:
    """Extract ALL NorthEuraLex data (all 1,016 concepts, all languages)."""
    cldf_dir = SOURCES / "northeuralex" / "cldf"
    if not cldf_dir.exists():
        print("  NorthEuraLex not found, skipping")
        return 0

    # Build language map: NEL Language_ID -> ISO code
    lang_map = {}
    for row in read_cldf_csv(cldf_dir / "languages.csv"):
        nel_id = row["ID"]
        iso = row.get("ISO639P3code", "")
        if not iso:
            # Many NEL IDs are already ISO 639-3
            iso = nel_id if len(nel_id) == 3 else ""
        if iso:
            lang_map[nel_id] = iso

    # Build parameter map: NEL Parameter_ID -> Concepticon_ID + gloss
    param_map = {}
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        pid = row["ID"]
        cid = row.get("Concepticon_ID", "")
        gloss = row.get("Concepticon_Gloss", row.get("Name", pid))
        param_map[pid] = (cid, gloss)

    # Process all forms
    processed = 0
    for row in read_cldf_csv(cldf_dir / "forms.csv"):
        lang_id = row.get("Language_ID", "")
        param_id = row.get("Parameter_ID", "")
        segments = row.get("Segments", "")

        iso = lang_map.get(lang_id)
        if not iso:
            continue

        ipa = segments_to_ipa(segments)
        if not ipa:
            continue

        word = row.get("Form", row.get("Value", ""))
        cid_info = param_map.get(param_id, ("", param_id))
        concept_id = cid_info[1] if cid_info[1] else param_id

        append_to_lexicon(lang_entries, iso, word, ipa, "northeuralex", concept_id)
        processed += 1

    print(f"  NorthEuraLex: {processed:,} entries from {len(set(lang_map.values()))} languages")
    return processed


def extract_wold(lang_entries: dict) -> int:
    """Extract ALL WOLD data (all concepts, all languages)."""
    cldf_dir = SOURCES / "wold" / "cldf"
    if not cldf_dir.exists():
        print("  WOLD not found, skipping")
        return 0

    # Build language map
    wold_lang_map = {}
    for row in read_cldf_csv(cldf_dir / "languages.csv"):
        wold_id = row["ID"]
        iso = row.get("ISO639P3code", "")
        if iso:
            wold_lang_map[wold_id] = iso

    # Build parameter map
    param_map = {}
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        pid = row["ID"]
        cid = row.get("Concepticon_ID", "")
        gloss = row.get("Concepticon_Gloss", row.get("Name", pid))
        param_map[pid] = (cid, gloss)

    # Process all forms
    processed = 0
    borrowing_count = 0

    for row in read_cldf_csv(cldf_dir / "forms.csv"):
        lang_id = row.get("Language_ID", "")
        param_id = row.get("Parameter_ID", "")
        segments = row.get("Segments", "")

        iso = wold_lang_map.get(lang_id)
        if not iso:
            continue

        ipa = segments_to_ipa(segments)
        if not ipa:
            continue

        word = row.get("Form", row.get("Value", ""))
        cid_info = param_map.get(param_id, ("", param_id))
        concept_id = cid_info[1] if cid_info[1] else param_id

        # Track borrowing status
        borrowed = row.get("Borrowed", "").strip()
        cognate_id = "-"
        if borrowed and borrowed not in ("0", ""):
            borrowing_count += 1

        append_to_lexicon(lang_entries, iso, word, ipa, "wold", concept_id, cognate_id)
        processed += 1

    print(f"  WOLD: {processed:,} entries from {len(set(wold_lang_map.values()))} languages ({borrowing_count:,} borrowings)")
    return processed


def extract_abvd(lang_entries: dict) -> int:
    """Extract ALL ABVD data (all concepts, Austronesian languages)."""
    cldf_dir = SOURCES / "abvd" / "cldf"
    if not cldf_dir.exists():
        print("  ABVD not found, skipping")
        return 0

    # Build language map — select Austronesian languages with ISO codes
    abvd_lang_map = {}
    for row in read_cldf_csv(cldf_dir / "languages.csv"):
        fam = row.get("Family", "")
        if fam != "Austronesian":
            continue
        iso = row.get("ISO639P3code", "")
        if not iso:
            continue
        abvd_lang_map[row["ID"]] = iso

    # Build parameter map
    param_map = {}
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        pid = row["ID"]
        cid = row.get("Concepticon_ID", "")
        gloss = row.get("Concepticon_Gloss", row.get("Name", pid))
        param_map[pid] = (cid, gloss)

    processed = 0
    cognate_count = 0

    forms_path = cldf_dir / "forms.csv"
    if not forms_path.exists():
        print("  ABVD forms.csv not found")
        return 0

    with open(forms_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            abvd_lang = row.get("Language_ID", "")
            iso = abvd_lang_map.get(abvd_lang)
            if not iso:
                continue

            param_id = row.get("Parameter_ID", "")
            form = row.get("Form", "").strip()
            if not form:
                continue

            cid_info = param_map.get(param_id, ("", param_id))
            concept_id = cid_info[1] if cid_info[1] else param_id

            # ABVD forms are orthographic — use as pseudo-IPA
            ipa = form_to_pseudo_ipa(form)
            if not ipa:
                continue

            cognacy = row.get("Cognacy", "").strip()
            cognate_id = "-"
            if cognacy:
                cog_num = cognacy.split(",")[0].strip()
                if cog_num:
                    cognate_id = f"abvd_{param_id}_{cog_num}"
                    cognate_count += 1

            append_to_lexicon(lang_entries, iso, form, ipa, "abvd", concept_id, cognate_id)
            processed += 1

    print(f"  ABVD: {processed:,} entries from {len(set(abvd_lang_map.values()))} languages ({cognate_count:,} with cognacy)")
    return processed


def extract_sinotibetan(lang_entries: dict) -> int:
    """Extract ALL sinotibetan data."""
    dump_path = SOURCES / "sinotibetan" / "sinotibetan_dump.tsv"
    if not dump_path.exists():
        dump_path = SOURCES / "sinotibetan" / "dumps" / "sinotibetan.tsv"
    if not dump_path.exists():
        print("  Sino-Tibetan dump not found, skipping")
        return 0

    # Doculect -> ISO code
    doculect_map = {
        "Old_Chinese": "och",
        "Japhug": "jya",
        "Tibetan_Written": "bod",
        "Old_Burmese": "obr",
        "Jingpho": "kac",
        "Lisu": "lis",
        "Naxi": "nxq",
        "Khaling": "klr",
        "Limbu": "lif",
        "Pumi_Lanping": "pmi",
        "Qiang_Mawo": "qxs",
        "Tujia": "tji",
        "Dulong": "duu",
        "Hakha": "cnh",
        "Bai_Jianchuan": "bca",
    }

    processed = 0
    cognate_count = 0

    with open(dump_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            doculect = row.get("DOCULECT", "")
            iso = doculect_map.get(doculect)
            if not iso:
                continue

            concept = row.get("CONCEPT", "").strip()
            ipa = row.get("IPA", "").strip()
            if not ipa or not concept:
                continue

            cogid = row.get("COGID", "").strip()
            cognate_id = "-"
            if cogid:
                cognate_id = f"st_{cogid}"
                cognate_count += 1

            append_to_lexicon(lang_entries, iso, concept, ipa, "sinotibetan", concept, cognate_id)
            processed += 1

    print(f"  Sino-Tibetan: {processed:,} entries ({cognate_count:,} with cognacy)")
    return processed


def write_lexicons(lang_entries: dict[str, list]) -> dict[str, int]:
    """Write per-language lexicon TSV files. Returns {iso: count}."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    for iso in sorted(lang_entries):
        entries = lang_entries[iso]
        if not entries:
            continue

        out_path = OUTPUT_DIR / f"{iso}.tsv"

        # If file already exists (e.g., from WikiPron), append new entries
        existing = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        existing.add((row[0], row[1]))

        mode = "a" if out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8", newline="") as f:
            if mode == "w":
                f.write(HEADER)
            new_count = 0
            for word, ipa, sca, source, concept_id, cognate_set_id in entries:
                if (word, ipa) not in existing:
                    f.write(f"{word}\t{ipa}\t{sca}\t{source}\t{concept_id}\t{cognate_set_id}\n")
                    existing.add((word, ipa))
                    new_count += 1

        total = len(existing)
        stats[iso] = total
        if new_count > 0 and total >= 100:
            pass  # Don't spam output for small languages

    return stats


def main():
    print("=" * 80)
    print("Full CLDF Extraction (All Concepts)")
    print("=" * 80)

    lang_entries: dict[str, list] = defaultdict(list)

    print("\nExtracting from CLDF sources...")
    total = 0

    for name, extractor in [
        ("NorthEuraLex", extract_northeuralex),
        ("WOLD", extract_wold),
        ("ABVD", extract_abvd),
        ("Sino-Tibetan", extract_sinotibetan),
    ]:
        print(f"\n  [{name}]")
        count = extractor(lang_entries)
        total += count

    print(f"\n{'=' * 80}")
    print(f"Writing lexicon files...")
    stats = write_lexicons(lang_entries)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  CLDF entries extracted: {total:,}")
    print(f"  Languages written:      {len(stats)}")
    total_written = sum(stats.values())
    print(f"  Total entries in files: {total_written:,}")

    # Write stats
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = METADATA_DIR / "cldf_stats.tsv"
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        f.write("ISO\tEntries\tSource\n")
        for iso, count in sorted(stats.items(), key=lambda x: -x[1]):
            f.write(f"{iso}\t{count}\tcldf\n")

    print(f"  Stats: {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
