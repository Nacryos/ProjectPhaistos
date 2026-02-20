"""Tests for CSV/TSV/COG ingester."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognate_pipeline.config.schema import ColumnMapping, SourceDef, SourceFormat
from cognate_pipeline.ingest.csv_ingester import CsvIngester


class TestCogIngester:
    """Test .cog format ingestion (Ugaritic-Hebrew cognate pairs)."""

    @pytest.fixture
    def cog_source(self) -> SourceDef:
        return SourceDef(
            name="uga_heb_test",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            license="Research / Fair Use",
            extra={"lang_a": "uga", "lang_b": "heb"},
        )

    def test_ingest_produces_lexemes(self, cog_source: SourceDef):
        ingester = CsvIngester(cog_source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) > 0

    def test_paired_structure(self, cog_source: SourceDef):
        """Each row should produce two lexemes (lang_a and lang_b)."""
        ingester = CsvIngester(cog_source)
        lexemes = list(ingester.ingest())
        # Should have pairs — roughly 2 lexemes per data row
        uga_lexemes = [l for l in lexemes if l.language_id == "uga"]
        heb_lexemes = [l for l in lexemes if l.language_id == "heb"]
        assert len(uga_lexemes) > 0
        assert len(heb_lexemes) > 0

    def test_pipe_separated_alternatives(self, cog_source: SourceDef):
        """Hebrew forms with | should be split into primary + alternatives."""
        ingester = CsvIngester(cog_source)
        lexemes = list(ingester.ingest())
        heb_with_alts = [l for l in lexemes if l.alternatives]
        # The file contains entries like "brr|brwr"
        assert len(heb_with_alts) > 0
        for l in heb_with_alts:
            assert l.language_id == "heb"

    def test_provenance_present(self, cog_source: SourceDef):
        ingester = CsvIngester(cog_source)
        lexemes = list(ingester.ingest())
        for l in lexemes[:5]:
            assert l.provenance is not None
            assert l.provenance.source_format == "cog"
            assert len(l.provenance.steps) >= 1

    def test_concept_ids_shared(self, cog_source: SourceDef):
        """Paired lexemes should share concept_id."""
        ingester = CsvIngester(cog_source)
        lexemes = list(ingester.ingest())
        # Group by concept_id
        by_concept: dict[str, list] = {}
        for l in lexemes:
            by_concept.setdefault(l.concept_id, []).append(l)
        # Most concepts should have exactly 2 lexemes (one per language)
        pair_counts = [len(v) for v in by_concept.values()]
        assert any(c == 2 for c in pair_counts)


class TestTsvIngester:
    def test_ingest_tsv(self, tmp_path: Path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text(
            "Language_ID\tForm\tParameter_ID\n"
            "eng\twater\twater\n"
            "deu\tWasser\twater\n"
            "fra\teau\twater\n"
        )
        source = SourceDef(
            name="test_tsv",
            path=tsv,
            format=SourceFormat.TSV,
        )
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) == 3
        assert lexemes[0].form == "water"
        assert lexemes[0].language_id == "eng"
        assert lexemes[0].concept_id == "water"

    def test_skip_empty_forms(self, tmp_path: Path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text(
            "Language_ID\tForm\tParameter_ID\n"
            "eng\t\twater\n"
            "eng\t_\tfire\n"
            "eng\thand\thand\n"
        )
        source = SourceDef(name="t", path=tsv, format=SourceFormat.TSV)
        lexemes = list(CsvIngester(source).ingest())
        assert len(lexemes) == 1
        assert lexemes[0].form == "hand"


class TestCsvIngester:
    def test_custom_column_mapping(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "lang,word,meaning\n"
            "english,fire,fire\n"
            "german,Feuer,fire\n"
        )
        source = SourceDef(
            name="custom",
            path=csv_file,
            format=SourceFormat.CSV,
            column_mapping=ColumnMapping(
                language="lang", form="word", concept="meaning"
            ),
        )
        lexemes = list(CsvIngester(source).ingest())
        assert len(lexemes) == 2
        assert lexemes[0].language_id == "english"
        assert lexemes[0].form == "fire"
