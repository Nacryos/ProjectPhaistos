"""Tests for Wiktionary JSONL ingester."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognate_pipeline.config.schema import SourceDef, SourceFormat
from cognate_pipeline.ingest.wiktionary_ingester import WiktionaryIngester


@pytest.fixture
def wikt_source(fixtures_dir: Path) -> SourceDef:
    return SourceDef(
        name="test_wikt",
        path=fixtures_dir / "sample_wiktionary.jsonl",
        format=SourceFormat.WIKTIONARY,
        license="CC-BY-SA-3.0",
    )


class TestWiktionaryIngester:
    def test_ingest_count(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        assert len(lexemes) == 10

    def test_ipa_extracted(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        water_en = [l for l in lexemes if l.form == "water" and l.language_id == "en"]
        assert len(water_en) == 1
        assert water_en[0].ipa_raw == "/ˈwɔːtə/"

    def test_concept_from_gloss(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        # First entry should have concept from first gloss
        assert lexemes[0].concept_id != ""

    def test_etymology_in_extra(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        house_en = [l for l in lexemes if l.form == "house"]
        assert len(house_en) == 1
        assert "etymology" in house_en[0].extra

    def test_language_code(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        lang_codes = {l.language_id for l in lexemes}
        assert "en" in lang_codes
        assert "de" in lang_codes
        assert "fr" in lang_codes

    def test_provenance(self, wikt_source: SourceDef):
        ingester = WiktionaryIngester(wikt_source)
        lexemes = list(ingester.ingest())
        for l in lexemes:
            assert l.provenance is not None
            assert l.provenance.source_format == "wiktionary"

    def test_handles_malformed_line(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text(
            '{"word": "good", "lang": "English", "lang_code": "en", "senses": [{"glosses": ["ok"]}]}\n'
            "not valid json\n"
            '{"word": "also_good", "lang": "German", "lang_code": "de", "senses": [{"glosses": ["ok"]}]}\n'
        )
        source = SourceDef(name="bad", path=f, format=SourceFormat.WIKTIONARY)
        lexemes = list(WiktionaryIngester(source).ingest())
        assert len(lexemes) == 2
