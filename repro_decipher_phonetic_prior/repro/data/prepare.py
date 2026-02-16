"""Deterministic dataset preparation and provenance logging."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets.registry import (
    Corpus,
    corpus_fingerprint,
    export_for_decipherunsegmented,
    get_corpus,
    list_corpora,
)
from repro.paths import ARTIFACTS, DATA_EXTERNAL, DATA_PREPARED, ROOT, THIRD_PARTY
from repro.third_party import load_lock
from repro.utils import sha256_file, utc_now_iso, write_json


EXPERIMENT_REQUIREMENTS = {
    "gothic": [
        "wiktionary_descendants_pg.tsv",
        "wiktionary_descendants_on.tsv",
        "wiktionary_descendants_oe.tsv",
    ],
    "iberian": [
        "rodriguez_ramos_2014_personal_names.tsv",
    ],
    "ugaritic": [],
}


def _candidate_sources(corpus_name: str) -> List[Tuple[str, Path]]:
    if corpus_name.startswith("validation_"):
        return [
            ("ancient-scripts-datasets", THIRD_PARTY / "ancient-scripts-datasets" / "data" / "validation"),
        ]
    if corpus_name == "gothic":
        return [
            ("ancient-scripts-datasets", THIRD_PARTY / "ancient-scripts-datasets" / "data" / "gothic"),
            ("DecipherUnsegmented", THIRD_PARTY / "DecipherUnsegmented" / "data"),
        ]
    if corpus_name == "ugaritic":
        return [
            ("NeuroDecipher", THIRD_PARTY / "NeuroDecipher" / "data"),
            ("ancient-scripts-datasets", THIRD_PARTY / "ancient-scripts-datasets" / "data" / "ugaritic"),
        ]
    if corpus_name == "iberian":
        return [
            ("DecipherUnsegmented", THIRD_PARTY / "DecipherUnsegmented" / "data"),
            ("ancient-scripts-datasets", THIRD_PARTY / "ancient-scripts-datasets" / "data" / "iberian"),
        ]
    return []


def _canonical_source(corpus_name: str) -> str:
    if corpus_name.startswith("validation_"):
        return "ancient-scripts-datasets"
    if corpus_name == "gothic":
        return "ancient-scripts-datasets"
    if corpus_name == "ugaritic":
        return "NeuroDecipher"
    if corpus_name == "iberian":
        return "DecipherUnsegmented"
    raise ValueError(f"Unknown corpus {corpus_name}")


def _repo_commit_map() -> Dict[str, Dict[str, str]]:
    lock = load_lock()
    out = {}
    for repo in lock.get("repos", []):
        out[repo["name"]] = {
            "commit": repo["commit"],
            "commit_date": repo["commit_date"],
            "url": repo["url"],
        }
    return out


def _copy_optional_assets(corpus: Corpus, out_dir: Path) -> List[str]:
    copied: List[str] = []
    if corpus.name == "gothic" and corpus.variant is None:
        src_root = THIRD_PARTY / "ancient-scripts-datasets" / "data" / "gothic"
        assets = ["segments.pkl", "got.pretrained.pth", "gotica.xml.zip"]
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            src = src_root / asset
            if src.exists():
                dst = assets_dir / asset
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied.append(str(dst))
    return copied


def prepare_one(corpus_name: str, variant: Optional[str], config: Dict[str, Any]) -> Dict[str, Any]:
    corpus = get_corpus(corpus_name, variant=variant)
    fingerprint = corpus_fingerprint(corpus, extra=config)
    variant_key = variant or "default"
    out_dir = DATA_PREPARED / corpus_name / variant_key / fingerprint
    exported = export_for_decipherunsegmented(corpus, out_dir, config)
    copied_assets = _copy_optional_assets(corpus, out_dir)

    manifest = {
        "corpus": corpus.name,
        "variant": corpus.variant,
        "fingerprint": fingerprint,
        "num_lost_lines": len(corpus.lost_text),
        "num_known_vocab": {k: len(v) for k, v in corpus.known_text.items()},
        "num_ground_truth_rows": len(corpus.ground_truth or []),
        "exported": exported,
        "copied_assets": copied_assets,
        "prepared_at": utc_now_iso(),
    }
    write_json(out_dir / "manifest.json", manifest)

    return {
        "name": corpus.name,
        "variant": corpus.variant,
        "fingerprint": fingerprint,
        "out_dir": str(out_dir),
        "canonical_source": _canonical_source(corpus_name),
        "candidate_sources": [
            {"repo": repo, "path": str(path), "exists": path.exists()} for repo, path in _candidate_sources(corpus_name)
        ],
        "manifest": manifest,
    }


def _missing_external_requirements(corpus_names: List[str]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for name in corpus_names:
        needed = EXPERIMENT_REQUIREMENTS.get(name, [])
        not_found = [fname for fname in needed if not (DATA_EXTERNAL / fname).exists()]
        if not_found:
            missing[name] = not_found
    return missing


def _external_checksums() -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    if not DATA_EXTERNAL.exists():
        return checksums
    for path in sorted(DATA_EXTERNAL.glob("**/*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums[str(path.relative_to(ROOT))] = sha256_file(path)
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic datasets for replication.")
    parser.add_argument(
        "--corpora",
        type=str,
        default=",".join(list_corpora()),
        help="Comma-separated corpus names.",
    )
    parser.add_argument(
        "--include-religious-variants",
        action="store_true",
        help="Prepare additional religious variants (not used by default runs).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if appendix external assets are missing.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-lines", type=int, default=0, help="Optional cap for quick debug preparation.")
    args = parser.parse_args()

    selected = [item.strip().lower() for item in args.corpora.split(",") if item.strip()]
    for name in selected:
        if name not in list_corpora():
            raise ValueError(f"Unknown corpus: {name}")

    prep_config = {
        "seed": args.seed,
        "max_lines": args.max_lines,
    }

    prepared: List[Dict[str, Any]] = []
    for name in selected:
        prepared.append(prepare_one(name, variant=None, config=prep_config))
        if args.include_religious_variants and not name.startswith("validation_"):
            prepared.append(prepare_one(name, variant="religious", config=prep_config))

    missing_external = _missing_external_requirements(selected)
    if args.strict and missing_external:
        missing_str = json.dumps(missing_external, indent=2)
        raise SystemExit(
            "Missing appendix/external assets required for full replication. "
            f"Run scripts/fetch_data.sh first.\n{missing_str}"
        )

    provenance = {
        "prepared_at": utc_now_iso(),
        "project_root": str(ROOT),
        "paper_pdf_used": str((Path("/Users/aaronbao/Downloads") / "DecipherUnsegmented (3).pdf")),
        "repos": _repo_commit_map(),
        "prepared": prepared,
        "missing_external_requirements": missing_external,
        "external_cache_checksums": _external_checksums(),
    }
    write_json(ARTIFACTS / "data_provenance.json", provenance)

    print(f"Prepared {len(prepared)} dataset variant(s).")
    print(f"Wrote provenance to {ARTIFACTS / 'data_provenance.json'}")


if __name__ == "__main__":
    main()
