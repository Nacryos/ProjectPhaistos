"""End-to-end tests for the 9 phylogenetic validation branches.

Parametrized over all branch TSV files. Each test verifies:
1. Ingestion produces expected number of lexemes
2. IPA transcription type is correctly assigned
3. No unknown SCA segments in IPA forms
4. Family-aware tagging produces cognate_inherited pairs
5. Scoring produces links above threshold
6. Minimum pair count is met
7. Clustering produces cognate sets
"""

from __future__ import annotations

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
from cognate_pipeline.ingest.models import TranscriptionType
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.normalise.models import NormalisedLexeme
from cognate_pipeline.normalise.sound_class import tokenize_ipa

VALIDATION_DIR = Path(r"C:\Users\alvin\ancient-scripts-datasets\data\validation")

# (branch_name, min_entries_after_null_skip)
BRANCHES = [
    ("germanic", 140),
    ("celtic", 110),
    ("balto_slavic", 110),
    ("indo_iranian", 100),
    ("italic", 30),       # Oscan/Umbrian have many gaps
    ("hellenic", 25),      # Mycenaean has many gaps
    ("semitic", 110),
    ("turkic", 110),
    ("uralic", 110),
]


def _make_source(branch_name: str) -> SourceDef:
    return SourceDef(
        name=branch_name,
        path=VALIDATION_DIR / f"{branch_name}.tsv",
        format=SourceFormat.TSV,
        license="Research / Fair Use",
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


@pytest.mark.parametrize("branch,min_entries", BRANCHES)
class TestValidationBranch:

    def test_ingest_count(self, branch: str, min_entries: int):
        """Branch TSV produces at least min_entries lexemes after null skip."""
        source = _make_source(branch)
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) >= min_entries, (
            f"{branch}: expected >= {min_entries} lexemes, got {len(lexemes)}"
        )

    def test_ipa_transcription_type(self, branch: str, min_entries: int):
        """All ingested entries with IPA should have TranscriptionType.IPA."""
        source = _make_source(branch)
        ingester = CsvIngester(source)
        for lex in ingester.ingest():
            if lex.phonetic_raw:
                assert lex.transcription_type == TranscriptionType.IPA, (
                    f"{branch}: '{lex.form}' has IPA '{lex.phonetic_raw}' "
                    f"but type={lex.transcription_type}"
                )

    def test_no_unknown_sca_segments(self, branch: str, min_entries: int):
        """No IPA form should produce '0' (unknown) in its SCA encoding."""
        normalised = _ingest_and_normalise(branch)
        unknowns = []
        for n in normalised:
            if "0" in n.sound_class:
                tokens = tokenize_ipa(n.phonetic_canonical or n.form)
                bad = [t for i, t in enumerate(tokens)
                       if i < len(n.sound_class) and n.sound_class[i] == "0"]
                unknowns.append((n.form, n.sound_class, bad))
        assert len(unknowns) == 0, (
            f"{branch}: {len(unknowns)} forms with unknown SCA segments: "
            + ", ".join(f"'{f}' -> '{sc}' (bad: {b})" for f, sc, b in unknowns[:5])
        )

    def test_family_aware_tagging(self, branch: str, min_entries: int):
        """Within-branch pairs should all be tagged cognate_inherited."""
        normalised = _ingest_and_normalise(branch)
        pairs = generate_candidates(normalised, family_aware=True)
        inherited = [p for p in pairs if p[2] == "cognate_inherited"]
        assert len(inherited) > 0, (
            f"{branch}: no cognate_inherited pairs found"
        )
        # All pairs should be inherited (same family)
        for a, b, rel in pairs:
            assert rel == "cognate_inherited", (
                f"{branch}: {a.language_id}-{b.language_id} tagged '{rel}', "
                f"expected 'cognate_inherited'"
            )

    def test_scoring_above_threshold(self, branch: str, min_entries: int):
        """Scoring produces links above 0.3 threshold."""
        normalised = _ingest_and_normalise(branch)
        pairs = generate_candidates(normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        assert len(links) > 0, (
            f"{branch}: no links above 0.3 threshold"
        )

    def test_minimum_pair_count(self, branch: str, min_entries: int):
        """Branch produces a reasonable number of candidate pairs."""
        normalised = _ingest_and_normalise(branch)
        pairs = generate_candidates(normalised, family_aware=True)
        # At minimum, each shared concept with 2+ languages produces pairs
        assert len(pairs) >= 10, (
            f"{branch}: only {len(pairs)} pairs, expected >= 10"
        )

    def test_clustering(self, branch: str, min_entries: int):
        """Clustering produces cognate sets from scored links."""
        normalised = _ingest_and_normalise(branch)
        pairs = generate_candidates(normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        if len(links) == 0:
            pytest.skip(f"{branch}: no links to cluster")
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) > 0, (
            f"{branch}: clustering produced no cognate sets"
        )
        for cs in sets:
            assert len(cs.members) >= 2


class TestNamesSubset:
    """End-to-end tests for the cross-family names validation subset.

    Unlike the single-family branches, names.tsv spans 30 languages across
    multiple families, so family-aware tagging produces both cognate_inherited
    and similarity_only pairs.
    """

    BRANCH = "names"
    MIN_ENTRIES = 140  # 160 non-null expected, allow some margin

    def test_ingest_count(self):
        """Names TSV produces at least MIN_ENTRIES lexemes after null skip."""
        source = _make_source(self.BRANCH)
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) >= self.MIN_ENTRIES, (
            f"names: expected >= {self.MIN_ENTRIES} lexemes, got {len(lexemes)}"
        )

    def test_ipa_transcription_type(self):
        """All ingested entries with IPA should have TranscriptionType.IPA."""
        source = _make_source(self.BRANCH)
        ingester = CsvIngester(source)
        for lex in ingester.ingest():
            if lex.phonetic_raw:
                assert lex.transcription_type == TranscriptionType.IPA, (
                    f"names: '{lex.form}' has IPA '{lex.phonetic_raw}' "
                    f"but type={lex.transcription_type}"
                )

    def test_no_unknown_sca_segments(self):
        """No IPA form should produce '0' (unknown) in its SCA encoding."""
        normalised = _ingest_and_normalise(self.BRANCH)
        unknowns = []
        for n in normalised:
            if "0" in n.sound_class:
                tokens = tokenize_ipa(n.phonetic_canonical or n.form)
                bad = [t for i, t in enumerate(tokens)
                       if i < len(n.sound_class) and n.sound_class[i] == "0"]
                unknowns.append((n.form, n.sound_class, bad))
        assert len(unknowns) == 0, (
            f"names: {len(unknowns)} forms with unknown SCA segments: "
            + ", ".join(f"'{f}' -> '{sc}' (bad: {b})" for f, sc, b in unknowns[:5])
        )

    def test_family_aware_tagging(self):
        """Cross-family names produce both inherited and similarity_only pairs."""
        normalised = _ingest_and_normalise(self.BRANCH)
        pairs = generate_candidates(normalised, family_aware=True)
        inherited = [p for p in pairs if p[2] == "cognate_inherited"]
        similarity = [p for p in pairs if p[2] == "similarity_only"]
        assert len(inherited) > 0, "names: no cognate_inherited pairs found"
        assert len(similarity) > 0, "names: no similarity_only pairs found"

    def test_scoring_above_threshold(self):
        """Scoring produces links above 0.3 threshold."""
        normalised = _ingest_and_normalise(self.BRANCH)
        pairs = generate_candidates(normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        assert len(links) > 0, "names: no links above 0.3 threshold"

    def test_minimum_pair_count(self):
        """Names produces a reasonable number of candidate pairs."""
        normalised = _ingest_and_normalise(self.BRANCH)
        pairs = generate_candidates(normalised, family_aware=True)
        assert len(pairs) >= 50, (
            f"names: only {len(pairs)} pairs, expected >= 50"
        )

    def test_clustering(self):
        """Clustering produces cognate sets from scored links."""
        normalised = _ingest_and_normalise(self.BRANCH)
        pairs = generate_candidates(normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        if len(links) == 0:
            pytest.skip("names: no links to cluster")
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) > 0, "names: clustering produced no cognate sets"
        for cs in sets:
            assert len(cs.members) >= 2
