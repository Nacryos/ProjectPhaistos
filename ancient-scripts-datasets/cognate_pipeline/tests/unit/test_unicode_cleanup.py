"""Tests for Unicode cleanup utilities."""

from __future__ import annotations

from cognate_pipeline.normalise.unicode_cleanup import (
    clean_whitespace,
    full_cleanup,
    normalize_unicode,
    strip_ipa_delimiters,
    strip_suprasegmentals,
)


class TestNormalizeUnicode:
    def test_nfc(self):
        # e + combining acute -> precomposed é
        composed = "\u00e9"
        decomposed = "e\u0301"
        assert normalize_unicode(decomposed, "NFC") == composed

    def test_nfkc(self):
        # Fullwidth A -> normal A
        assert normalize_unicode("\uff21", "NFKC") == "A"


class TestStripIpaDelimiters:
    def test_slashes(self):
        assert strip_ipa_delimiters("/ˈwɔːtə/") == "ˈwɔːtə"

    def test_brackets(self):
        assert strip_ipa_delimiters("[ˈwɔːtə]") == "ˈwɔːtə"

    def test_no_delimiters(self):
        assert strip_ipa_delimiters("wɔːtə") == "wɔːtə"

    def test_empty(self):
        assert strip_ipa_delimiters("") == ""


class TestStripSuprasegmentals:
    def test_stress_marks(self):
        result = strip_suprasegmentals("ˈwɔːtə")
        assert "ˈ" not in result

    def test_tone_diacritics(self):
        # High tone on a
        result = strip_suprasegmentals("a\u0301")
        assert "\u0301" not in result


class TestCleanWhitespace:
    def test_multiple_spaces(self):
        assert clean_whitespace("a  b   c") == "a b c"

    def test_leading_trailing(self):
        assert clean_whitespace("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        assert clean_whitespace("a\t\nb") == "a b"


class TestFullCleanup:
    def test_combined(self):
        result = full_cleanup("/ˈwɔːtə/", strip_supra=True)
        assert "/" not in result
        assert "ˈ" not in result
        assert result == "wɔːtə"

    def test_default_no_supra_strip(self):
        result = full_cleanup("/ˈwɔːtə/")
        assert "ˈ" in result  # Stress preserved by default

    def test_empty(self):
        assert full_cleanup("") == ""
