"""Tests for CLDF ingester."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognate_pipeline.config.schema import SourceDef, SourceFormat


class TestCldfIngester:
    @pytest.fixture
    def cldf_source(self, sample_cldf_dir: Path) -> SourceDef:
        return SourceDef(
            name="test_cldf",
            path=sample_cldf_dir,
            format=SourceFormat.CLDF,
            license="CC-BY-4.0",
        )

    def test_ingest_count(self, cldf_source: SourceDef):
        pycldf = pytest.importorskip("pycldf")
        from cognate_pipeline.ingest.cldf_ingester import CldfIngester

        ingester = CldfIngester(cldf_source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) == 15  # 3 languages × 5 concepts

    def test_glottocode_mapped(self, cldf_source: SourceDef):
        pycldf = pytest.importorskip("pycldf")
        from cognate_pipeline.ingest.cldf_ingester import CldfIngester

        ingester = CldfIngester(cldf_source)
        lexemes = list(ingester.ingest())
        uga = [l for l in lexemes if l.language_id == "lang_uga"]
        assert all(l.glottocode == "ugar1238" for l in uga)

    def test_segments_to_ipa(self, cldf_source: SourceDef):
        pycldf = pytest.importorskip("pycldf")
        from cognate_pipeline.ingest.cldf_ingester import CldfIngester

        ingester = CldfIngester(cldf_source)
        lexemes = list(ingester.ingest())
        # All forms have Segments in our fixture
        for l in lexemes:
            assert l.ipa_raw, f"Missing IPA for {l.id}"

    def test_provenance(self, cldf_source: SourceDef):
        pycldf = pytest.importorskip("pycldf")
        from cognate_pipeline.ingest.cldf_ingester import CldfIngester

        ingester = CldfIngester(cldf_source)
        lexemes = list(ingester.ingest())
        for l in lexemes:
            assert l.provenance is not None
            assert l.provenance.source_format == "cldf"
