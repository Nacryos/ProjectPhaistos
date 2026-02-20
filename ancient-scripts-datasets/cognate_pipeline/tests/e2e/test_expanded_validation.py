"""End-to-end tests for expanded validation branches (CLDF-derived).

Tests all new and expanded TSV files generated from NorthEuraLex,
WOLD, ABVD, and sinotibetan CLDF repositories.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from cognate_pipeline.cognate.baseline_levenshtein import BaselineLevenshtein
from cognate_pipeline.cognate.candidate_gen import generate_candidates
from cognate_pipeline.cognate.clustering import cluster_links
from cognate_pipeline.config.schema import (
    ClusteringAlgorithm,
    ColumnMapping,
    NormalisationConfig,
    SourceDef,
    SourceFormat,
)
from cognate_pipeline.ingest.csv_ingester import CsvIngester
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.normalise.models import NormalisedLexeme
from cognate_pipeline.normalise.sound_class import tokenize_ipa

VALIDATION_DIR = Path(r"C:\Users\alvin\ancient-scripts-datasets\data\validation")

# (branch_name, min_entries, min_languages)
EXPANDED_BRANCHES = [
    # Expanded existing families
    ("germanic_expanded", 600, 8),
    ("celtic_expanded", 200, 3),
    ("balto_slavic_expanded", 800, 10),
    ("indo_iranian_expanded", 500, 7),
    ("italic_expanded", 500, 7),
    ("hellenic_expanded", 100, 2),
    ("semitic_expanded", 150, 2),
    ("turkic_expanded", 600, 8),
    ("uralic_expanded", 1500, 20),
    # New families
    ("albanian", 80, 1),
    ("armenian", 80, 1),
    ("dravidian", 300, 4),
    ("kartvelian", 80, 1),
    ("austronesian", 1000, 15),
    ("sino_tibetan", 200, 5),
    ("mongolic", 200, 3),
    ("tungusic", 200, 3),
    ("japonic", 80, 1),
    ("koreanic", 80, 1),
    ("northeast_caucasian", 500, 6),
    ("northwest_caucasian", 150, 2),
    ("eskimo_aleut", 200, 3),
    ("isolates", 300, 4),
    ("afroasiatic_berber", 80, 1),
    ("afroasiatic_chadic", 80, 1),
    ("afroasiatic_cushitic", 150, 2),
    ("niger_congo_bantu", 80, 1),
    ("tai_kadai", 80, 1),
    ("austroasiatic", 150, 2),
    ("mayan", 150, 2),
    ("quechuan", 80, 1),
    ("uto_aztecan", 80, 1),
    ("hmong_mien", 80, 1),
    ("chukotko_kamchatkan", 150, 2),
    ("yukaghir", 150, 2),
    ("saharan", 80, 1),
]


def _make_source(branch_name: str) -> SourceDef:
    return SourceDef(
        name=branch_name,
        path=VALIDATION_DIR / f"{branch_name}.tsv",
        format=SourceFormat.TSV,
        license="CC-BY / Research",
        column_mapping=ColumnMapping(
            language="Language_ID",
            form="Form",
            concept="Parameter_ID",
            ipa="IPA",
            glottocode="Glottocode",
        ),
    )


def _ingest_and_normalise(branch_name: str) -> list[NormalisedLexeme]:
    source = _make_source(branch_name)
    ingester = CsvIngester(source)
    config = NormalisationConfig(
        ipa_backend_priority=["attested"],
        transliteration_passthrough=False,
    )
    normaliser = IpaNormaliser(config)
    return [normaliser.normalise(r) for r in ingester.ingest()]


@pytest.mark.parametrize(
    "branch,min_entries,min_langs",
    EXPANDED_BRANCHES,
    ids=[b[0] for b in EXPANDED_BRANCHES],
)
class TestExpandedBranch:
    """Parametrized tests for all expanded/new validation branches."""

    def test_file_exists(self, branch: str, min_entries: int, min_langs: int):
        """TSV file exists."""
        path = VALIDATION_DIR / f"{branch}.tsv"
        assert path.exists(), f"{branch}.tsv not found"

    def test_ingest_count(self, branch: str, min_entries: int, min_langs: int):
        """Branch produces at least min_entries lexemes."""
        source = _make_source(branch)
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) >= min_entries, (
            f"{branch}: expected >= {min_entries} lexemes, got {len(lexemes)}"
        )

    def test_language_count(self, branch: str, min_entries: int, min_langs: int):
        """Branch contains at least min_langs distinct languages."""
        source = _make_source(branch)
        ingester = CsvIngester(source)
        lang_ids = set(lex.language_id for lex in ingester.ingest())
        assert len(lang_ids) >= min_langs, (
            f"{branch}: expected >= {min_langs} languages, got {len(lang_ids)}: {lang_ids}"
        )

    def test_no_duplicate_lang_concept(self, branch: str, min_entries: int, min_langs: int):
        """No duplicate (Language_ID, Parameter_ID) within a TSV."""
        path = VALIDATION_DIR / f"{branch}.tsv"
        seen = set()
        dupes = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row["Language_ID"], row["Parameter_ID"])
                if key in seen:
                    dupes.append(key)
                seen.add(key)
        assert len(dupes) == 0, (
            f"{branch}: {len(dupes)} duplicate (lang, concept) pairs: {dupes[:5]}"
        )

    def test_scoring_produces_links(self, branch: str, min_entries: int, min_langs: int):
        """Scoring produces links above 0.3 threshold (multi-lang branches)."""
        if min_langs < 2:
            pytest.skip(f"{branch}: single-language branch, no pairs possible")
        normalised = _ingest_and_normalise(branch)
        pairs = generate_candidates(normalised, family_aware=True)
        if len(pairs) == 0:
            pytest.skip(f"{branch}: no candidate pairs generated")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        assert len(links) > 0, f"{branch}: no links above 0.3 threshold"


class TestGlobalCoverage:
    """Coverage assertions across all expanded TSV files."""

    def _count_all_entries(self) -> tuple[int, set[str], set[str]]:
        total = 0
        all_langs = set()
        all_concepts = set()
        for f in os.listdir(VALIDATION_DIR):
            if not f.endswith(".tsv"):
                continue
            # Skip metadata files
            if f in ("concepts.tsv", "concepts_expanded.tsv", "languages.tsv",
                      "names_pairs.tsv"):
                continue
            path = VALIDATION_DIR / f
            with open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    if row.get("IPA", "_") == "_" or row.get("Form", "_") == "_":
                        continue
                    total += 1
                    all_langs.add(row.get("Language_ID", ""))
                    all_concepts.add(row.get("Parameter_ID", ""))
        return total, all_langs, all_concepts

    def test_minimum_200_languages(self):
        """At least 200 unique languages across all validation TSVs."""
        _, all_langs, _ = self._count_all_entries()
        assert len(all_langs) >= 200, (
            f"Expected >= 200 languages, got {len(all_langs)}"
        )

    def test_minimum_100_concepts(self):
        """At least 100 unique concepts across all expanded TSVs."""
        _, _, all_concepts = self._count_all_entries()
        assert len(all_concepts) >= 100, (
            f"Expected >= 100 concepts, got {len(all_concepts)}"
        )

    def test_minimum_15000_entries(self):
        """At least 15,000 total word entries across all TSVs."""
        total, _, _ = self._count_all_entries()
        assert total >= 15000, (
            f"Expected >= 15,000 entries, got {total:,}"
        )

    def test_concepts_expanded_file(self):
        """concepts_expanded.tsv exists and has > 100 concepts."""
        path = VALIDATION_DIR / "concepts_expanded.tsv"
        assert path.exists(), "concepts_expanded.tsv not found"
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) >= 100, (
            f"Expected >= 100 concepts in expanded list, got {len(rows)}"
        )

    def test_languages_tsv_file(self):
        """languages.tsv exists and has > 150 languages."""
        path = VALIDATION_DIR / "languages.tsv"
        assert path.exists(), "languages.tsv not found"
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) >= 150, (
            f"Expected >= 150 languages in master list, got {len(rows)}"
        )

    def test_all_glottocodes_valid_format(self):
        """All Glottocodes match the 8-char format (xxxx1234)."""
        path = VALIDATION_DIR / "languages.tsv"
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                gc = row.get("Glottocode", "")
                if gc:
                    assert len(gc) == 8 and gc[:4].isalpha() and gc[4:].isdigit(), (
                        f"Invalid Glottocode: {gc} for {row.get('Language_ID')}"
                    )


class TestCrossFamilyExpanded:
    """Test that loading multiple branches produces both relationship types."""

    def test_cross_family_pairs(self):
        """Loading two different families produces similarity_only pairs."""
        germanic = _ingest_and_normalise("germanic_expanded")
        italic = _ingest_and_normalise("italic_expanded")
        combined = germanic + italic
        pairs = generate_candidates(combined, family_aware=True)
        inherited = [p for p in pairs if p[2] == "cognate_inherited"]
        similarity = [p for p in pairs if p[2] == "similarity_only"]
        assert len(inherited) > 0, "No cognate_inherited pairs found"
        assert len(similarity) > 0, "No similarity_only pairs found"
