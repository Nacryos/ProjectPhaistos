"""End-to-end tests for the stratified validation datasets.

Validates the output of scripts/build_validation_sets.py: file existence,
schema correctness, label/phylo_dist/timespan validity, and structural
constraints (L1 pairs share a sub-branch, L4/false positives cross families,
religious files contain only religious concepts, etc.).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Ensure cognate_pipeline is importable (for RELIGIOUS_ALL)
_PIPELINE_SRC = (
    Path(__file__).resolve().parent.parent.parent / "src"
)
sys.path.insert(0, str(_PIPELINE_SRC))

# Paths relative to the ancient-scripts-datasets repo
# test is at: ancient-scripts-datasets/cognate_pipeline/tests/e2e/test_validation_sets.py
# parent chain: e2e -> tests -> cognate_pipeline -> ancient-scripts-datasets
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VALIDATION_DIR = REPO_ROOT / "data" / "training" / "validation"
FAMILY_MAP_PATH = (
    REPO_ROOT
    / "cognate_pipeline"
    / "src"
    / "cognate_pipeline"
    / "cognate"
    / "family_map.json"
)

# Expected column header
EXPECTED_FIELDS = [
    "Lang_A", "Word_A", "IPA_A", "SCA_A",
    "Lang_B", "Word_B", "IPA_B", "SCA_B",
    "Concept_ID", "Label", "Phylo_Dist", "Timespan", "Score", "Source",
]

VALID_LABELS = {"true_cognate", "false_positive", "true_negative", "borrowing"}
VALID_PHYLO_DIST = {"L1", "L2", "L3", "L4"}
VALID_TIMESPANS = {"ancient_ancient", "ancient_modern", "medieval_modern", "modern_modern"}

TOP_FAMILIES = [
    "germanic", "italic", "balto_slavic", "indo_iranian", "hellenic",
    "celtic", "uralic", "turkic", "sino_tibetan", "austronesian",
    "semitic", "dravidian", "japonic", "koreanic", "kartvelian",
]

RELIGIOUS_SUBDOMAINS = [
    "core_religious",
    "supernatural",
    "moral_ethical",
    "ritual_ceremony",
    "religious_verbs",
    "cosmic_spiritual",
    "sacred_places",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file and return a list of row dicts."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_tsv_header(path: Path) -> list[str]:
    """Read only the header of a TSV file."""
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        return next(reader)


@pytest.fixture(scope="module")
def family_map() -> dict[str, str]:
    """Load family_map.json."""
    with FAMILY_MAP_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def phylo_tree() -> dict:
    """Load phylo_tree.json."""
    tree_path = VALIDATION_DIR / "phylo_tree.json"
    with tree_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _collect_isos_from_tree(node) -> set[str]:
    """Recursively collect all ISO codes from a tree node."""
    if isinstance(node, list):
        return set(node)
    if isinstance(node, str):
        return {node}
    if isinstance(node, dict):
        isos: set[str] = set()
        for v in node.values():
            isos |= _collect_isos_from_tree(v)
        return isos
    return set()


def _find_iso_in_tree(tree: dict, iso: str) -> list[str] | None:
    """Find the path to an ISO code in the tree. Returns list of keys."""
    def _walk(node, prefix):
        if isinstance(node, list):
            if iso in node:
                return prefix
        elif isinstance(node, str):
            if node == iso:
                return prefix
        elif isinstance(node, dict):
            for key, child in node.items():
                result = _walk(child, prefix + [key])
                if result is not None:
                    return result
        return None
    return _walk(tree, [])


# ---------------------------------------------------------------------------
# Core file existence tests
# ---------------------------------------------------------------------------

CORE_FILES = [
    "phylo_tree.json",
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
    "validation_stats.tsv",
]


@pytest.mark.parametrize("filename", CORE_FILES)
def test_core_file_exists_and_nonempty(filename):
    """All core output files exist and are non-empty."""
    path = VALIDATION_DIR / filename
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 0, f"Empty: {path}"


@pytest.mark.parametrize("family", TOP_FAMILIES)
def test_per_family_file_exists(family):
    """Per-family validation files exist."""
    path = VALIDATION_DIR / "per_family" / f"{family}.tsv"
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 0, f"Empty: {path}"


# ---------------------------------------------------------------------------
# Header format tests
# ---------------------------------------------------------------------------

PAIR_FILES = [
    "true_cognates_L1.tsv",
    "true_cognates_L2.tsv",
    "true_cognates_L3.tsv",
    "false_positives.tsv",
    "true_negatives.tsv",
    "borrowings.tsv",
]


@pytest.mark.parametrize("filename", PAIR_FILES)
def test_header_format(filename):
    """Header matches expected 14-column schema."""
    path = VALIDATION_DIR / filename
    header = _read_tsv_header(path)
    assert header == EXPECTED_FIELDS, f"Header mismatch in {filename}: {header}"


# ---------------------------------------------------------------------------
# Label, Phylo_Dist, Timespan validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", PAIR_FILES)
def test_labels_valid(filename):
    """All Label values are one of the valid set."""
    path = VALIDATION_DIR / filename
    rows = _read_tsv(path)
    labels = {r["Label"] for r in rows}
    invalid = labels - VALID_LABELS
    assert not invalid, f"Invalid labels in {filename}: {invalid}"


@pytest.mark.parametrize("filename", PAIR_FILES)
def test_phylo_dist_valid(filename):
    """All Phylo_Dist values are L1-L4."""
    path = VALIDATION_DIR / filename
    rows = _read_tsv(path)
    dists = {r["Phylo_Dist"] for r in rows}
    invalid = dists - VALID_PHYLO_DIST
    assert not invalid, f"Invalid phylo_dist in {filename}: {invalid}"


@pytest.mark.parametrize("filename", PAIR_FILES)
def test_timespan_valid(filename):
    """All Timespan values are valid."""
    path = VALIDATION_DIR / filename
    rows = _read_tsv(path)
    timespans = {r["Timespan"] for r in rows}
    invalid = timespans - VALID_TIMESPANS
    assert not invalid, f"Invalid timespans in {filename}: {invalid}"


# ---------------------------------------------------------------------------
# Structural constraint tests
# ---------------------------------------------------------------------------


def test_l1_same_subbranch(phylo_tree):
    """L1 pairs: both languages should be in the same sub-branch of the tree."""
    rows = _read_tsv(VALIDATION_DIR / "true_cognates_L1.tsv")
    # Sample 200 pairs to keep the test fast
    sample = rows[:200]
    violations = 0
    for row in sample:
        path_a = _find_iso_in_tree(phylo_tree, row["Lang_A"])
        path_b = _find_iso_in_tree(phylo_tree, row["Lang_B"])
        if path_a is not None and path_b is not None:
            if path_a != path_b:
                violations += 1
    # Allow some violations (languages resolved from family_map may not be in tree)
    assert violations < len(sample) * 0.1, (
        f"Too many L1 violations: {violations}/{len(sample)}"
    )


def test_false_positives_cross_family(family_map):
    """False positive pairs should be from different families."""
    rows = _read_tsv(VALIDATION_DIR / "false_positives.tsv")
    sample = rows[:200]
    cross_family = 0
    for row in sample:
        fam_a = family_map.get(row["Lang_A"], "unknown_a")
        fam_b = family_map.get(row["Lang_B"], "unknown_b")
        if fam_a != fam_b:
            cross_family += 1
    # At least 80% should be cross-family
    assert cross_family >= len(sample) * 0.8, (
        f"Only {cross_family}/{len(sample)} false positives are cross-family"
    )


def test_per_family_languages_belong(family_map):
    """Per-family files should primarily contain languages from that family."""
    for fam in TOP_FAMILIES[:5]:  # Test a sample of families
        path = VALIDATION_DIR / "per_family" / f"{fam}.tsv"
        if not path.exists():
            continue
        rows = _read_tsv(path)
        sample = rows[:200]
        matching = 0
        for row in sample:
            fam_a = family_map.get(row["Lang_A"], "")
            fam_b = family_map.get(row["Lang_B"], "")
            # At least one language should belong to this family
            # (cross-family false positives will have one language from another family)
            if fam_a == fam or fam_b == fam:
                matching += 1
            elif fam == "balto_slavic" and (fam_a == "slavic" or fam_b == "slavic"):
                matching += 1
        # At least 70% should have at least one language from the family
        assert matching >= len(sample) * 0.7, (
            f"Only {matching}/{len(sample)} pairs in {fam}.tsv "
            f"have a language from {fam}"
        )


# ---------------------------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------------------------


def test_no_duplicates_l1():
    """No duplicate pairs in L1 file."""
    rows = _read_tsv(VALIDATION_DIR / "true_cognates_L1.tsv")
    keys = set()
    dups = 0
    for row in rows:
        key = (row["Lang_A"], row["Word_A"], row["Lang_B"], row["Word_B"], row["Concept_ID"])
        if key in keys:
            dups += 1
        keys.add(key)
    # Allow very small number of duplicates (from different sources)
    assert dups < len(rows) * 0.01, f"Too many duplicates in L1: {dups}/{len(rows)}"


# ---------------------------------------------------------------------------
# Minimum pair count tests
# ---------------------------------------------------------------------------


def test_minimum_pairs_l1():
    """L1 file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "true_cognates_L1.tsv")
    assert len(rows) >= 1_000, f"L1 only has {len(rows)} pairs"


def test_minimum_pairs_l2():
    """L2 file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "true_cognates_L2.tsv")
    assert len(rows) >= 1_000, f"L2 only has {len(rows)} pairs"


def test_minimum_pairs_l3():
    """L3 file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "true_cognates_L3.tsv")
    assert len(rows) >= 1_000, f"L3 only has {len(rows)} pairs"


def test_minimum_pairs_false_positives():
    """False positives file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "false_positives.tsv")
    assert len(rows) >= 1_000, f"False positives only has {len(rows)} pairs"


def test_minimum_pairs_true_negatives():
    """True negatives file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "true_negatives.tsv")
    assert len(rows) >= 1_000, f"True negatives only has {len(rows)} pairs"


def test_minimum_pairs_borrowings():
    """Borrowings file should have at least 1,000 pairs."""
    rows = _read_tsv(VALIDATION_DIR / "borrowings.tsv")
    assert len(rows) >= 1_000, f"Borrowings only has {len(rows)} pairs"


# ---------------------------------------------------------------------------
# Religious subset tests (new religious/ directory structure)
# ---------------------------------------------------------------------------

RELIGIOUS_DIR = VALIDATION_DIR / "religious"

RELIGIOUS_CORE_FILES = [
    "all_pairs.tsv",
    "true_cognates.tsv",
    "false_positives.tsv",
    "borrowings.tsv",
]


@pytest.mark.parametrize("filename", RELIGIOUS_CORE_FILES)
def test_religious_core_file_exists(filename):
    """Religious core files exist in the religious/ directory."""
    path = RELIGIOUS_DIR / filename
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 0, f"Empty: {path}"


@pytest.mark.parametrize("filename", RELIGIOUS_CORE_FILES)
def test_religious_core_file_header(filename):
    """Religious core files have the correct header."""
    path = RELIGIOUS_DIR / filename
    header = _read_tsv_header(path)
    assert header == EXPECTED_FIELDS, f"Header mismatch in religious/{filename}: {header}"


def test_religious_true_cognates_label_only():
    """religious/true_cognates.tsv should only contain true_cognate labels."""
    rows = _read_tsv(RELIGIOUS_DIR / "true_cognates.tsv")
    labels = {r["Label"] for r in rows}
    assert labels == {"true_cognate"}, (
        f"Expected only true_cognate labels, got: {labels}"
    )


def test_religious_false_positives_label_only():
    """religious/false_positives.tsv should only contain false_positive labels."""
    rows = _read_tsv(RELIGIOUS_DIR / "false_positives.tsv")
    labels = {r["Label"] for r in rows}
    assert labels == {"false_positive"}, (
        f"Expected only false_positive labels, got: {labels}"
    )


def test_religious_no_compound_concept_ids():
    """No compound concept IDs (containing '/') in religious files."""
    for filename in RELIGIOUS_CORE_FILES:
        path = RELIGIOUS_DIR / filename
        rows = _read_tsv(path)
        compound = [r["Concept_ID"] for r in rows if "/" in r["Concept_ID"]]
        assert len(compound) == 0, (
            f"{len(compound)} compound concept IDs in religious/{filename}: "
            f"{compound[:5]}"
        )


def test_religious_concept_diversity():
    """Religious all_pairs.tsv should have at least 20 unique concepts."""
    rows = _read_tsv(RELIGIOUS_DIR / "all_pairs.tsv")
    concepts = {r["Concept_ID"] for r in rows}
    assert len(concepts) >= 20, (
        f"Religious all_pairs has only {len(concepts)} unique concepts: {concepts}"
    )


@pytest.mark.parametrize("subdomain", RELIGIOUS_SUBDOMAINS)
def test_religious_subdomain_file_exists(subdomain):
    """Sub-domain files exist and are non-empty in the religious/ directory."""
    path = RELIGIOUS_DIR / f"{subdomain}.tsv"
    assert path.exists(), f"Missing: {path}"
    assert path.stat().st_size > 0, f"Empty: {path}"


@pytest.mark.parametrize("subdomain", RELIGIOUS_SUBDOMAINS)
def test_religious_subdomain_header(subdomain):
    """Sub-domain files have the correct header."""
    path = RELIGIOUS_DIR / f"{subdomain}.tsv"
    header = _read_tsv_header(path)
    assert header == EXPECTED_FIELDS, (
        f"Header mismatch in religious/{subdomain}.tsv: {header}"
    )


def test_religious_by_family_dir_exists():
    """religious/by_family/ directory exists with at least some family files."""
    by_family_dir = RELIGIOUS_DIR / "by_family"
    assert by_family_dir.exists(), f"Missing: {by_family_dir}"
    tsv_files = list(by_family_dir.glob("*.tsv"))
    assert len(tsv_files) >= 3, (
        f"Expected at least 3 family files, got {len(tsv_files)}"
    )


def test_old_religious_pairs_removed():
    """Old religious_pairs.tsv should no longer exist at the top level."""
    old_path = VALIDATION_DIR / "religious_pairs.tsv"
    assert not old_path.exists(), (
        f"Old file still exists: {old_path}"
    )


def test_old_religious_by_family_removed():
    """Old religious_by_family/ directory should no longer exist."""
    old_dir = VALIDATION_DIR / "religious_by_family"
    assert not old_dir.exists(), (
        f"Old directory still exists: {old_dir}"
    )


# ---------------------------------------------------------------------------
# Timespan tests
# ---------------------------------------------------------------------------


def test_at_least_3_timespans_populated():
    """At least 3 of the 4 timespan categories should have data."""
    populated = 0
    for ts in ["ancient_ancient", "ancient_modern", "medieval_modern", "modern_modern"]:
        path = VALIDATION_DIR / f"timespan_{ts}.tsv"
        if path.exists():
            rows = _read_tsv(path)
            if len(rows) > 0:
                populated += 1
    assert populated >= 3, f"Only {populated} timespan categories populated"


# ---------------------------------------------------------------------------
# Phylo tree structure
# ---------------------------------------------------------------------------


def test_phylo_tree_valid_json():
    """phylo_tree.json is valid JSON with expected top-level families."""
    tree_path = VALIDATION_DIR / "phylo_tree.json"
    with tree_path.open(encoding="utf-8") as fh:
        tree = json.load(fh)
    assert isinstance(tree, dict)
    # Should have at least a few top-level families
    expected_families = {"indo_european", "uralic", "turkic", "sino_tibetan"}
    assert expected_families.issubset(set(tree.keys())), (
        f"Missing families. Have: {set(tree.keys())}"
    )


def test_phylo_tree_contains_languages(phylo_tree):
    """Phylo tree should contain at least 100 unique ISO codes."""
    isos = _collect_isos_from_tree(phylo_tree)
    assert len(isos) >= 100, f"Tree only has {len(isos)} languages"


# ---------------------------------------------------------------------------
# Stats file
# ---------------------------------------------------------------------------


def test_stats_file_has_content():
    """validation_stats.tsv should have meaningful content."""
    path = VALIDATION_DIR / "validation_stats.tsv"
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    assert len(rows) >= 10, f"Stats file only has {len(rows)} rows"
    # Check header
    assert rows[0] == ["Category", "Subset", "Count"]


# ---------------------------------------------------------------------------
# Language profile tests
# ---------------------------------------------------------------------------

PROFILES_DIR = REPO_ROOT / "data" / "training" / "language_profiles"


def test_language_profiles_dir_exists():
    """language_profiles/ directory should exist."""
    assert PROFILES_DIR.exists(), f"Missing: {PROFILES_DIR}"


def test_language_profiles_count():
    """Should have 1,110 .md profile files (one per language)."""
    md_files = list(PROFILES_DIR.glob("*.md"))
    assert len(md_files) == 1110, (
        f"Expected 1110 profile files, got {len(md_files)}"
    )


@pytest.mark.parametrize("iso", ["eng", "lat", "fin"])
def test_language_profile_structure(iso):
    """Spot-check that key sections exist in profile markdown."""
    path = PROFILES_DIR / f"{iso}.md"
    assert path.exists(), f"Missing profile: {path}"
    content = path.read_text(encoding="utf-8")

    # Must have the main heading with ISO code
    assert f"(`{iso}`)" in content, f"Missing ISO in heading for {iso}"

    # Must have all expected section headers
    for section in [
        "## Overview",
        "## Lexicon Summary",
        "## Cognate Pair Participation",
        "## Timespan Distribution",
        "## Family-Internal Pairs",
        "## Religious Domain",
        "## Top Partner Languages",
    ]:
        assert section in content, f"Missing section '{section}' in {iso}.md"


@pytest.mark.parametrize("iso", ["eng", "lat", "fin"])
def test_language_profile_has_lexicon_data(iso):
    """Spot-check languages should have non-zero lexicon stats."""
    path = PROFILES_DIR / f"{iso}.md"
    content = path.read_text(encoding="utf-8")
    # These well-known languages should have entries
    assert "| Total entries | 0 |" not in content, (
        f"{iso}.md shows 0 total entries"
    )


@pytest.mark.parametrize("iso", ["eng", "lat", "fin"])
def test_language_profile_has_validation_pairs(iso):
    """Spot-check languages should participate in validation pairs."""
    path = PROFILES_DIR / f"{iso}.md"
    content = path.read_text(encoding="utf-8")
    # At least one cognate level should be non-zero
    has_pairs = False
    for label in ["True Cognates L1", "True Cognates L2", "True Cognates L3"]:
        # Find the line for this label and check it's not "| ... | 0 |"
        for line in content.splitlines():
            if label in line and "| 0 |" not in line:
                has_pairs = True
                break
    assert has_pairs, f"{iso}.md has no cognate pairs"
