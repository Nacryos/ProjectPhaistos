"""Stable corpus registry for all data sources used in this replication."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Corpus:
    name: str
    variant: Optional[str]
    lost_text: List[str]
    known_text: Dict[str, List[str]]
    metadata: Dict[str, Any]
    splits: Dict[str, List[int]]
    ground_truth: Optional[List[Dict[str, Any]]] = None


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_ASD_DATA = _THIRD_PARTY / "ancient-scripts-datasets" / "data"
_VALIDATION_DIR = _ASD_DATA / "validation"

_CANONICAL_SOURCES = {
    "gothic": "ancient-scripts-datasets",
    "ugaritic": "NeuroDecipher",
    "iberian": "DecipherUnsegmented",
}

_BASE_CORPORA = ("gothic", "ugaritic", "iberian")


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf8").splitlines()


def _tokenize_plain_text(lines: Iterable[str]) -> List[str]:
    token_lines: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lower()
        line = re.sub(r"[^\w\s\-þðƕȝōēæáéíóúïöü]+", " ", line)
        tokens = [tok for tok in line.split() if tok]
        if tokens:
            token_lines.append(" ".join(tokens))
    return token_lines


def _hash_to_bucket(value: str, mod: int) -> int:
    return int(hashlib.sha256(value.encode("utf8")).hexdigest(), 16) % mod


def _deterministic_splits(lost_text: List[str]) -> Dict[str, List[int]]:
    train: List[int] = []
    dev: List[int] = []
    test: List[int] = []
    for idx, line in enumerate(lost_text):
        bucket = _hash_to_bucket(f"{idx}:{line}", 10)
        if bucket <= 7:
            train.append(idx)
        elif bucket == 8:
            dev.append(idx)
        else:
            test.append(idx)
    return {"train": train, "dev": dev, "test": test}


def _read_tsv_with_comments(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf8") as fin:
        header: Optional[List[str]] = None
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if header is None:
                header = stripped.split("\t")
                continue
            values = stripped.split("\t")
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            row = dict(zip(header, values))
            rows.append(row)
    return rows


def _validation_branches() -> List[str]:
    if not _VALIDATION_DIR.exists():
        return []
    names = []
    for path in sorted(_VALIDATION_DIR.glob("*.tsv")):
        if path.name == "concepts.tsv":
            continue
        names.append(path.stem.lower())
    return names


def _validation_corpora() -> List[str]:
    return [f"validation_{branch}" for branch in _validation_branches()]


def _validation_path(branch: str) -> Path:
    return _VALIDATION_DIR / f"{branch}.tsv"


def _clean_surface(row: Dict[str, str]) -> str:
    ipa = (row.get("IPA") or "").strip()
    form = (row.get("Form") or "").strip()
    if ipa and ipa != "_":
        return ipa
    if form and form != "_":
        return form
    return ""


def _read_validation_rows(branch: str) -> List[Dict[str, str]]:
    path = _validation_path(branch)
    if not path.exists():
        raise ValueError(f"Validation branch {branch!r} not found at {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            lang = (row.get("Language_ID") or "").strip().lower()
            concept = (row.get("Parameter_ID") or "").strip()
            token = _clean_surface(row)
            if not lang or not concept or not token:
                continue
            rows.append(
                {
                    "language": lang,
                    "concept": concept,
                    "token": token,
                    "form": (row.get("Form") or "").strip(),
                    "ipa": (row.get("IPA") or "").strip(),
                    "glottocode": (row.get("Glottocode") or "").strip(),
                }
            )
    return rows


def _parse_validation_variant(
    variant: Optional[str],
    languages: Sequence[str],
) -> Tuple[str, List[str], Dict[str, str]]:
    if not languages:
        raise ValueError("No languages found in validation branch.")

    sorted_langs = sorted(set(languages))
    meta: Dict[str, str] = {}
    if variant is None:
        lost = sorted_langs[0]
        known = [lang for lang in sorted_langs if lang != lost]
        meta["selection"] = "default"
        return lost, known, meta

    if variant.startswith("lang:"):
        lost = variant.split(":", 1)[1].strip().lower()
        if lost not in sorted_langs:
            raise ValueError(f"Unknown validation language {lost!r}, choices: {sorted_langs}")
        known = [lang for lang in sorted_langs if lang != lost]
        meta["selection"] = "lang"
        return lost, known, meta

    if variant.startswith("pair:"):
        parts = [p.strip().lower() for p in variant.split(":")]
        if len(parts) != 3:
            raise ValueError("Pair variant must be in format pair:<lost_lang>:<known_lang>")
        _, lost, known_lang = parts
        if lost not in sorted_langs:
            raise ValueError(f"Unknown lost language {lost!r}, choices: {sorted_langs}")
        if known_lang not in sorted_langs:
            raise ValueError(f"Unknown known language {known_lang!r}, choices: {sorted_langs}")
        if known_lang == lost:
            raise ValueError("pair variant requires different lost/known languages.")
        meta["selection"] = "pair"
        return lost, [known_lang], meta

    raise ValueError(
        "Unsupported validation variant. Use one of: "
        "None (default), lang:<lost_lang>, pair:<lost_lang>:<known_lang>."
    )


def _build_validation_corpus(branch: str, variant: Optional[str]) -> Corpus:
    rows = _read_validation_rows(branch)
    languages = sorted({row["language"] for row in rows})
    lost_lang, known_langs, selection_meta = _parse_validation_variant(variant, languages)

    by_concept_lang: Dict[Tuple[str, str], List[str]] = {}
    for row in rows:
        by_concept_lang.setdefault((row["concept"], row["language"]), []).append(row["token"])

    lost_records = [row for row in rows if row["language"] == lost_lang]
    lost_text = [row["token"] for row in lost_records]

    known_text: Dict[str, List[str]] = {}
    for lang in known_langs:
        known_text[lang] = sorted({row["token"] for row in rows if row["language"] == lang})

    union_vocab: List[str] = []
    for vocab in known_text.values():
        union_vocab.extend(vocab)
    known_text["known_union"] = sorted(set(union_vocab))

    gt: List[Dict[str, Any]] = []
    for row in lost_records:
        candidates: List[str] = []
        for lang in known_langs:
            candidates.extend(by_concept_lang.get((row["concept"], lang), []))
        gt.append(
            {
                "lost": row["token"],
                "known": sorted(set(candidates)),
                "concept_id": row["concept"],
                "lost_language": lost_lang,
                "known_languages": "|".join(known_langs),
                "branch": branch,
            }
        )

    metadata = {
        "canonical_source": "ancient-scripts-datasets",
        "source_paths": {
            "validation_branch": str(_validation_path(branch)),
            "validation_concepts": str(_VALIDATION_DIR / "concepts.tsv"),
        },
        "validation_branch": branch,
        "available_languages": languages,
        "lost_language": lost_lang,
        "known_languages": known_langs,
        "representation": "IPA_first_fallback_form",
        **selection_meta,
    }
    return Corpus(
        name=f"validation_{branch}",
        variant=variant,
        lost_text=lost_text,
        known_text=known_text,
        metadata=metadata,
        splits=_deterministic_splits(lost_text),
        ground_truth=gt,
    )


def _build_gothic_default() -> Corpus:
    source = _THIRD_PARTY / "ancient-scripts-datasets" / "data" / "gothic"
    gotica_path = source / "gotica.txt"
    lost_text = _tokenize_plain_text(_read_lines(gotica_path))
    metadata = {
        "canonical_source": _CANONICAL_SOURCES["gothic"],
        "source_paths": {
            "gotica": str(gotica_path),
            "pretrained_embedding": str(source / "got.pretrained.pth"),
            "segments": str(source / "segments.pkl"),
        },
        "notes": "Known-language vocabularies (PG/ON/OE) are external and prepared separately.",
    }
    return Corpus(
        name="gothic",
        variant=None,
        lost_text=lost_text,
        known_text={},
        metadata=metadata,
        splits=_deterministic_splits(lost_text),
        ground_truth=None,
    )


def _build_gothic_religious() -> Corpus:
    path = _THIRD_PARTY / "ancient-scripts-datasets" / "data" / "religious_terms" / "gothic_religious.tsv"
    rows = _read_tsv_with_comments(path)
    lost = [row.get("gothic_word", "").strip() for row in rows if row.get("gothic_word", "").strip()]
    english = [row.get("english_meaning", "").strip() for row in rows if row.get("gothic_word", "").strip()]
    gt = [
        {
            "lost": row.get("gothic_word", "").strip(),
            "known": row.get("english_meaning", "").strip(),
            "category": row.get("category", "").strip(),
            "subcategory": row.get("subcategory", "").strip(),
        }
        for row in rows
        if row.get("gothic_word", "").strip()
    ]
    metadata = {
        "canonical_source": "ancient-scripts-datasets",
        "source_paths": {"religious_terms": str(path)},
        "is_religious_subset": True,
        "unused_by_default": True,
    }
    return Corpus(
        name="gothic",
        variant="religious",
        lost_text=lost,
        known_text={"english_gloss": english},
        metadata=metadata,
        splits=_deterministic_splits(lost),
        ground_truth=gt,
    )


def _build_ugaritic_default() -> Corpus:
    path = _THIRD_PARTY / "NeuroDecipher" / "data" / "uga-heb.no_spe.cog"
    lost_text: List[str] = []
    hebrew_vocab: List[str] = []
    gt: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            lost = (row.get("uga-no_spe") or "").strip()
            known_raw = (row.get("heb-no_spe") or "").strip()
            if not lost or not known_raw or known_raw == "_":
                continue
            known_candidates = [tok.strip() for tok in known_raw.split("|") if tok.strip()]
            if not known_candidates:
                continue
            lost_text.append(lost)
            hebrew_vocab.extend(known_candidates)
            gt.append({"lost": lost, "known": known_candidates})
    metadata = {
        "canonical_source": _CANONICAL_SOURCES["ugaritic"],
        "source_paths": {"cognates": str(path)},
        "known_column": "heb-no_spe",
        "lost_column": "uga-no_spe",
    }
    return Corpus(
        name="ugaritic",
        variant=None,
        lost_text=lost_text,
        known_text={"hebrew": sorted(set(hebrew_vocab))},
        metadata=metadata,
        splits=_deterministic_splits(lost_text),
        ground_truth=gt,
    )


def _build_ugaritic_religious() -> Corpus:
    path = _THIRD_PARTY / "ancient-scripts-datasets" / "data" / "religious_terms" / "ugaritic_hebrew_religious.tsv"
    rows = _read_tsv_with_comments(path)
    lost: List[str] = []
    heb: List[str] = []
    gt: List[Dict[str, Any]] = []
    for row in rows:
        lost_form = row.get("ugaritic_form", "").strip()
        known_raw = row.get("hebrew_cognate", "").strip()
        if not lost_form:
            continue
        candidates = [tok.strip() for tok in known_raw.split("|") if tok.strip()]
        if not candidates:
            candidates = [known_raw] if known_raw else []
        lost.append(lost_form)
        heb.extend(candidates)
        gt.append(
            {
                "lost": lost_form,
                "known": candidates,
                "category": row.get("category", "").strip(),
                "subcategory": row.get("subcategory", "").strip(),
            }
        )
    metadata = {
        "canonical_source": "ancient-scripts-datasets",
        "source_paths": {"religious_terms": str(path)},
        "is_religious_subset": True,
        "unused_by_default": True,
    }
    return Corpus(
        name="ugaritic",
        variant="religious",
        lost_text=lost,
        known_text={"hebrew": sorted(set(heb))},
        metadata=metadata,
        splits=_deterministic_splits(lost),
        ground_truth=gt,
    )


def _build_iberian_default() -> Corpus:
    path = _THIRD_PARTY / "DecipherUnsegmented" / "data" / "iberian.csv"
    lost: List[str] = []
    with path.open("r", encoding="utf8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            raw = (row.get("cleaned") or "").strip()
            if not raw:
                continue
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    content = " ".join(str(x) for x in parsed if str(x).strip())
                else:
                    content = str(parsed)
            except (ValueError, SyntaxError):
                content = raw
            content = content.strip()
            if content:
                lost.append(content)
    metadata = {
        "canonical_source": _CANONICAL_SOURCES["iberian"],
        "source_paths": {"iberian_csv": str(path)},
        "notes": "Known vocab for personal-name experiment comes from Rodriguez Ramos (2014).",
    }
    return Corpus(
        name="iberian",
        variant=None,
        lost_text=lost,
        known_text={},
        metadata=metadata,
        splits=_deterministic_splits(lost),
        ground_truth=None,
    )


def _build_iberian_religious() -> Corpus:
    path = _THIRD_PARTY / "ancient-scripts-datasets" / "data" / "religious_terms" / "iberian_religious.tsv"
    rows = _read_tsv_with_comments(path)
    # The header in this file currently starts with "3category".
    first_col = next(iter(rows[0].keys())) if rows else "category"
    lost: List[str] = []
    known: List[str] = []
    gt: List[Dict[str, Any]] = []
    for row in rows:
        element = (row.get("element") or "").strip()
        proposed = (row.get("proposed_meaning") or "").strip()
        if not element:
            continue
        lost.append(element)
        if proposed:
            known.append(proposed)
        gt.append(
            {
                "lost": element,
                "known": proposed,
                "category": (row.get("category") or row.get(first_col) or "").strip(),
                "subcategory": row.get("subcategory", "").strip(),
            }
        )
    metadata = {
        "canonical_source": "ancient-scripts-datasets",
        "source_paths": {"religious_terms": str(path)},
        "is_religious_subset": True,
        "unused_by_default": True,
    }
    return Corpus(
        name="iberian",
        variant="religious",
        lost_text=lost,
        known_text={"proposed_meaning": sorted(set(known))},
        metadata=metadata,
        splits=_deterministic_splits(lost),
        ground_truth=gt,
    )


_BUILDERS = {
    ("gothic", None): _build_gothic_default,
    ("gothic", "religious"): _build_gothic_religious,
    ("ugaritic", None): _build_ugaritic_default,
    ("ugaritic", "religious"): _build_ugaritic_religious,
    ("iberian", None): _build_iberian_default,
    ("iberian", "religious"): _build_iberian_religious,
}


def list_corpora() -> List[str]:
    return list(_BASE_CORPORA) + _validation_corpora()


def get_corpus(name: str, variant: Optional[str] = None) -> Corpus:
    lowered = name.lower()
    if lowered.startswith("validation_"):
        branch = lowered.replace("validation_", "", 1)
        if variant == "religious":
            raise ValueError("Religious variant is not defined for validation corpora.")
        return _build_validation_corpus(branch, variant)

    key = (lowered, variant)
    if key not in _BUILDERS:
        raise ValueError(f"Unsupported corpus/variant combination: {name!r}, {variant!r}")
    corpus = _BUILDERS[key]()
    return corpus


def list_validation_languages(corpus_name: str) -> List[str]:
    lowered = corpus_name.lower()
    if not lowered.startswith("validation_"):
        raise ValueError("list_validation_languages expects a validation corpus name (validation_<branch>).")
    branch = lowered.replace("validation_", "", 1)
    rows = _read_validation_rows(branch)
    return sorted({row["language"] for row in rows})


def export_for_decipherunsegmented(
    corpus: Corpus,
    out_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    config = config or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lost_path = out / "lost.txt"
    lost_path.write_text("\n".join(corpus.lost_text) + "\n", encoding="utf8")

    known_paths: Dict[str, str] = {}
    for known_name, vocab in corpus.known_text.items():
        kp = out / f"known_{known_name}.txt"
        kp.write_text("\n".join(sorted(set(vocab))) + "\n", encoding="utf8")
        known_paths[known_name] = str(kp)

    split_path = out / "splits.json"
    split_path.write_text(json.dumps(corpus.splits, indent=2) + "\n", encoding="utf8")

    meta = dict(corpus.metadata)
    meta["name"] = corpus.name
    meta["variant"] = corpus.variant
    meta["config"] = config
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf8")

    gt_path: Optional[Path] = None
    if corpus.ground_truth:
        gt_path = out / "ground_truth.tsv"
        all_fields = sorted({k for row in corpus.ground_truth for k in row.keys()})
        with gt_path.open("w", encoding="utf8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=all_fields, delimiter="\t")
            writer.writeheader()
            for row in corpus.ground_truth:
                writer.writerow(row)

    exported = {
        "lost_text": str(lost_path),
        "splits": str(split_path),
        "metadata": str(meta_path),
    }
    if known_paths:
        exported["known_text"] = json.dumps(known_paths, sort_keys=True)
    if gt_path is not None:
        exported["ground_truth"] = str(gt_path)
    return exported


def corpus_fingerprint(corpus: Corpus, extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "corpus": asdict(corpus),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()[:16]
