"""Tests for language resolver."""

from __future__ import annotations

from cognate_pipeline.ingest.language_resolver import LanguageResolver


class TestLanguageResolver:
    def test_glottocode_passthrough(self):
        resolver = LanguageResolver()
        assert resolver.resolve("ugar1238") == "ugar1238"
        assert resolver.resolve("hebr1245") == "hebr1245"

    def test_hardcoded_ancient_languages(self):
        resolver = LanguageResolver()
        assert resolver.resolve("uga") == "ugar1238"
        assert resolver.resolve("heb") == "hebr1245"
        assert resolver.resolve("got") == "goth1244"
        assert resolver.resolve("akk") == "akka1240"
        assert resolver.resolve("lat") == "lati1261"

    def test_case_insensitive(self):
        resolver = LanguageResolver()
        assert resolver.resolve("UGA") == "ugar1238"
        assert resolver.resolve("Heb") == "hebr1245"

    def test_unknown_returns_empty(self):
        resolver = LanguageResolver()
        assert resolver.resolve("xyz_unknown") == ""

    def test_empty_returns_empty(self):
        resolver = LanguageResolver()
        assert resolver.resolve("") == ""
        assert resolver.resolve("  ") == ""
