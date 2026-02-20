#!/usr/bin/env python3
"""Audit CLDF repositories for language/concept coverage.

Scans each cloned CLDF repo under sources/ and produces a coverage matrix
showing which target languages and concepts are available in which repo.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"

REPOS = {
    "northeuralex": SOURCES_DIR / "northeuralex" / "cldf",
    "ids": SOURCES_DIR / "ids" / "cldf",
    "abvd": SOURCES_DIR / "abvd" / "cldf",
    "wold": SOURCES_DIR / "wold" / "cldf",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit_repo(name: str, cldf_dir: Path) -> dict:
    result = {"name": name, "exists": cldf_dir.exists()}
    if not result["exists"]:
        return result

    langs = read_csv(cldf_dir / "languages.csv")
    params = read_csv(cldf_dir / "parameters.csv")

    # Count forms with/without IPA segments
    forms_path = cldf_dir / "forms.csv"
    total_forms = 0
    forms_with_segments = 0
    forms_with_cognacy = 0
    lang_ids = set()
    param_ids = set()

    if forms_path.exists():
        with open(forms_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                total_forms += 1
                if row.get("Segments", "").strip():
                    forms_with_segments += 1
                if row.get("Cognacy", "").strip():
                    forms_with_cognacy += 1
                lang_ids.add(row.get("Language_ID", ""))
                param_ids.add(row.get("Parameter_ID", ""))

    # Extract families
    families = {}
    for lang in langs:
        fam = lang.get("Family", "Unknown")
        families.setdefault(fam, []).append(lang.get("ID", ""))

    # Extract Concepticon IDs
    concepticon_ids = set()
    for p in params:
        cid = p.get("Concepticon_ID", "").strip()
        if cid:
            concepticon_ids.add(cid)

    result.update({
        "languages": len(langs),
        "parameters": len(params),
        "total_forms": total_forms,
        "forms_with_segments": forms_with_segments,
        "forms_with_cognacy": forms_with_cognacy,
        "segment_pct": 100 * forms_with_segments / max(total_forms, 1),
        "cognacy_pct": 100 * forms_with_cognacy / max(total_forms, 1),
        "lang_ids_in_forms": len(lang_ids),
        "param_ids_in_forms": len(param_ids),
        "families": families,
        "concepticon_ids": concepticon_ids,
    })
    return result


def main():
    print("=" * 80)
    print("CLDF Repository Audit")
    print("=" * 80)

    results = {}
    for name, cldf_dir in REPOS.items():
        print(f"\nAuditing {name}...")
        results[name] = audit_repo(name, cldf_dir)
        r = results[name]
        if not r["exists"]:
            print(f"  NOT FOUND: {cldf_dir}")
            continue
        print(f"  Languages: {r['languages']}")
        print(f"  Parameters (concepts): {r['parameters']}")
        print(f"  Total forms: {r['total_forms']:,}")
        print(f"  Forms with IPA segments: {r['forms_with_segments']:,} ({r['segment_pct']:.1f}%)")
        print(f"  Forms with cognacy: {r['forms_with_cognacy']:,} ({r['cognacy_pct']:.1f}%)")
        print(f"  Families: {len(r['families'])}")
        for fam, ids in sorted(r["families"].items(), key=lambda x: -len(x[1])):
            print(f"    {fam} ({len(ids)}): {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")

    # Concepticon overlap
    print("\n" + "=" * 80)
    print("Concepticon ID Overlap (concepts shared across repos)")
    print("=" * 80)
    repo_names = [n for n in REPOS if results[n]["exists"]]
    all_cids = set()
    for name in repo_names:
        cids = results[name].get("concepticon_ids", set())
        all_cids |= cids
        print(f"  {name}: {len(cids)} Concepticon IDs")

    # Find shared concepts
    shared_in_2plus = set()
    shared_in_3plus = set()
    for cid in all_cids:
        count = sum(1 for n in repo_names if cid in results[n].get("concepticon_ids", set()))
        if count >= 2:
            shared_in_2plus.add(cid)
        if count >= 3:
            shared_in_3plus.add(cid)

    print(f"\n  Shared in 2+ repos: {len(shared_in_2plus)}")
    print(f"  Shared in 3+ repos: {len(shared_in_3plus)}")

    # sinotibetan special case
    st_path = SOURCES_DIR / "sinotibetan" / "dumps" / "sinotibetan.tsv"
    if st_path.exists():
        print(f"\n{'=' * 80}")
        print("Sino-Tibetan dump (non-CLDF)")
        print("=" * 80)
        doculects = set()
        concepts = set()
        total = 0
        with open(st_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total += 1
                doculects.add(row.get("DOCULECT", ""))
                concepts.add(row.get("CONCEPT", ""))
        print(f"  Total entries: {total:,}")
        print(f"  Doculects: {len(doculects)}")
        print(f"  Concepts: {len(concepts)}")
        print(f"  Doculects: {sorted(doculects)}")


if __name__ == "__main__":
    main()
