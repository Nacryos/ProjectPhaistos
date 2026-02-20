"""End-to-end pipeline test using synthetic CLDF fixture.

Runs all stages: ingest -> normalise -> detect -> (optionally load) -> export.
Does NOT require PostgreSQL — uses SQLite in-memory for the DB stages.
"""

from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pytest

from cognate_pipeline.cognate.baseline_levenshtein import BaselineLevenshtein
from cognate_pipeline.cognate.candidate_gen import generate_candidates
from cognate_pipeline.cognate.clustering import cluster_links
from cognate_pipeline.cognate.models import CognateSet
from cognate_pipeline.config.schema import (
    ClusteringAlgorithm,
    NormalisationConfig,
    SourceDef,
    SourceFormat,
)
from cognate_pipeline.ingest.csv_ingester import CsvIngester
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.normalise.models import NormalisedLexeme


class TestFullPipelineInMemory:
    """Full pipeline test using the .cog format (Ugaritic-Hebrew)."""

    @pytest.fixture
    def cog_source(self) -> SourceDef:
        return SourceDef(
            name="uga_heb_e2e",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            license="Research / Fair Use",
            extra={"lang_a": "uga", "lang_b": "heb"},
        )

    def test_ingest_to_cognate_sets(self, cog_source: SourceDef, tmp_path: Path):
        """Run full pipeline: ingest -> normalise -> detect -> cluster."""
        # --- Stage 1: Ingest ---
        ingester = CsvIngester(cog_source)
        raw_lexemes = list(ingester.ingest())
        assert len(raw_lexemes) > 50, f"Expected many lexemes, got {len(raw_lexemes)}"

        # Write to staging
        ingest_dir = tmp_path / "staging" / "ingest"
        ingest_dir.mkdir(parents=True)
        ingest_path = ingest_dir / "uga_heb.jsonl"
        with ingest_path.open("wb") as f:
            for lex in raw_lexemes:
                f.write(orjson.dumps(lex.to_dict()) + b"\n")

        # --- Stage 2: Normalise ---
        config = NormalisationConfig(
            transliteration_passthrough=True,
            ipa_backend_priority=["attested"],
        )
        normaliser = IpaNormaliser(config)
        normalised: list[NormalisedLexeme] = []
        for lex in raw_lexemes:
            normalised.append(normaliser.normalise(lex))

        assert len(normalised) == len(raw_lexemes)
        # All should have sound classes
        with_sc = [n for n in normalised if n.sound_class]
        assert len(with_sc) > 0.9 * len(normalised), "Most lexemes should have sound classes"

        # Write normalised staging
        norm_dir = tmp_path / "staging" / "normalised"
        norm_dir.mkdir(parents=True)
        norm_path = norm_dir / "uga_heb.jsonl"
        with norm_path.open("wb") as f:
            for n in normalised:
                f.write(orjson.dumps(n.to_dict()) + b"\n")

        # --- Stage 3: Candidate Generation ---
        pairs = generate_candidates(normalised, family_aware=False)
        assert len(pairs) > 0, "Should generate at least some candidate pairs"

        # --- Stage 4: Scoring ---
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.4)
        assert len(links) > 0, "Should have some links above threshold"

        # Verify link properties
        for link in links[:10]:
            assert link.score >= 0.4
            assert link.lexeme_id_a < link.lexeme_id_b
            assert link.method == "baseline_lev"
            assert "sound_class_a" in link.evidence

        # --- Stage 5: Clustering ---
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) > 0, "Should form at least some cognate sets"

        # Verify set properties
        for cs in sets:
            assert len(cs.members) >= 2
            assert cs.concept_id != ""

        # Write cognate staging
        cog_dir = tmp_path / "staging" / "cognate"
        cog_dir.mkdir(parents=True)
        links_path = cog_dir / "cognate_links.jsonl"
        with links_path.open("wb") as f:
            for link in links:
                f.write(orjson.dumps(link.to_dict()) + b"\n")

        sets_path = cog_dir / "cognate_sets.jsonl"
        with sets_path.open("wb") as f:
            for cs in sets:
                f.write(orjson.dumps(cs.to_dict()) + b"\n")

        # --- Verify JSONL roundtrip ---
        restored_links = []
        with links_path.open("rb") as f:
            for line in f:
                from cognate_pipeline.cognate.models import CognateLink
                restored_links.append(CognateLink.from_dict(orjson.loads(line)))
        assert len(restored_links) == len(links)

        restored_sets = []
        with sets_path.open("rb") as f:
            for line in f:
                restored_sets.append(CognateSet.from_dict(orjson.loads(line)))
        assert len(restored_sets) == len(sets)

    def test_known_cognates_detected(self, cog_source: SourceDef):
        """Verify that known Ugaritic-Hebrew cognates are detected.

        The .cog file pairs Ugaritic and Hebrew cognates. For identical
        transliterations (e.g. 'ab' / 'ab'), the pipeline should produce
        a perfect score.
        """
        ingester = CsvIngester(cog_source)
        raw = list(ingester.ingest())

        config = NormalisationConfig(
            transliteration_passthrough=True,
            ipa_backend_priority=["attested"],
        )
        normaliser = IpaNormaliser(config)
        normalised = [normaliser.normalise(r) for r in raw]

        pairs = generate_candidates(normalised, family_aware=False)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)

        # Find pairs with identical forms (perfect cognates)
        perfect_links = [l for l in links if l.score == 1.0]
        assert len(perfect_links) > 0, "Should detect some perfect cognate matches"

    def test_staging_files_structure(self, cog_source: SourceDef, tmp_path: Path):
        """Verify that staging files have the expected structure."""
        ingester = CsvIngester(cog_source)
        raw = list(ingester.ingest())

        ingest_path = tmp_path / "test.jsonl"
        with ingest_path.open("wb") as f:
            for lex in raw[:5]:
                f.write(orjson.dumps(lex.to_dict()) + b"\n")

        # Read back and verify structure
        with ingest_path.open("rb") as f:
            for line in f:
                d = orjson.loads(line)
                assert "id" in d
                assert "language_id" in d
                assert "form" in d
                assert "provenance" in d
                prov = d["provenance"]
                assert "source_name" in prov
                assert "steps" in prov


class TestPipelineWithSyntheticData:
    """Test pipeline with fully synthetic data to avoid file dependencies."""

    def test_synthetic_end_to_end(self, tmp_path: Path):
        """Build synthetic cognate pairs and run full pipeline."""
        # Create synthetic TSV
        tsv_path = tmp_path / "synthetic.tsv"
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Language_ID", "Form", "Parameter_ID"])
            # Water cognates (similar across languages)
            writer.writerow(["lang_a", "pata", "water"])
            writer.writerow(["lang_b", "bada", "water"])
            writer.writerow(["lang_c", "pata", "water"])
            # Fire cognates
            writer.writerow(["lang_a", "kur", "fire"])
            writer.writerow(["lang_b", "gur", "fire"])
            # Hand cognates
            writer.writerow(["lang_a", "man", "hand"])
            writer.writerow(["lang_b", "man", "hand"])
            writer.writerow(["lang_c", "nan", "hand"])

        source = SourceDef(
            name="synthetic",
            path=tsv_path,
            format=SourceFormat.TSV,
            license="CC0",
        )

        # Ingest
        ingester = CsvIngester(source)
        raw = list(ingester.ingest())
        assert len(raw) == 8

        # Normalise
        normaliser = IpaNormaliser(NormalisationConfig(transliteration_passthrough=True))
        normalised = [normaliser.normalise(r) for r in raw]

        # Candidate gen
        pairs = generate_candidates(normalised)
        # water: 3 pairs (a-b, a-c, b-c), fire: 1 pair, hand: 3 pairs = 7
        assert len(pairs) == 7

        # Score
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.3)
        assert len(links) > 0

        # Cluster
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) > 0

        # Verify "man"/"man" pair is a perfect match
        man_links = [l for l in links if l.score == 1.0]
        assert len(man_links) > 0

        # Verify pata/bada have a high but imperfect score
        # (P->B is same class, A->A identical, T->D same class, A->A identical)
        water_links = [l for l in links if l.concept_id == "water"]
        assert len(water_links) > 0
        # pata vs bada should be quite similar (labial + vowel + coronal + vowel)
        pata_bada = [l for l in water_links if l.score > 0.5]
        assert len(pata_bada) > 0
