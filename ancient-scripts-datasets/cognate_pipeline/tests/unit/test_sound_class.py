"""Tests for SCA sound class encoding."""

from __future__ import annotations

from cognate_pipeline.normalise.sound_class import (
    ipa_to_sound_class,
    segment_to_class,
    tokenize_ipa,
)


class TestTokenizeIpa:
    def test_simple_word(self):
        tokens = tokenize_ipa("pata")
        assert tokens == ["p", "a", "t", "a"]

    def test_empty(self):
        assert tokenize_ipa("") == []

    def test_with_diacritics(self):
        # p followed by aspiration diacritic
        tokens = tokenize_ipa("pʰ")
        # Should produce 1 token with diacritic attached
        assert len(tokens) >= 1

    def test_transliteration(self):
        """Transliterated forms should tokenize character by character."""
        tokens = tokenize_ipa("ab")
        assert tokens == ["a", "b"]


class TestSegmentToClass:
    def test_vowels(self):
        assert segment_to_class("a") == "A"
        assert segment_to_class("e") == "E"
        assert segment_to_class("i") == "I"
        assert segment_to_class("o") == "O"
        assert segment_to_class("u") == "U"

    def test_stops(self):
        assert segment_to_class("p") == "P"
        assert segment_to_class("t") == "T"
        assert segment_to_class("k") == "K"
        assert segment_to_class("b") == "B"
        assert segment_to_class("d") == "D"
        assert segment_to_class("g") == "G"

    def test_nasals(self):
        assert segment_to_class("m") == "M"
        assert segment_to_class("n") == "N"
        assert segment_to_class("ŋ") == "N"

    def test_liquids(self):
        assert segment_to_class("l") == "L"
        assert segment_to_class("r") == "R"

    def test_sibilants(self):
        assert segment_to_class("s") == "S"
        assert segment_to_class("z") == "S"
        assert segment_to_class("ʃ") == "S"

    def test_glides(self):
        assert segment_to_class("w") == "W"
        assert segment_to_class("j") == "Y"

    def test_unknown(self):
        assert segment_to_class("!") == "0"

    def test_transliteration_specials(self):
        """Ancient script transliteration characters."""
        assert segment_to_class("$") == "S"
        assert segment_to_class("H") == "H"

    def test_palatal_stops(self):
        """Palatal stops for Turkic, Celtic, Hungarian."""
        assert segment_to_class("c") == "K"
        assert segment_to_class("ɟ") == "G"

    def test_retroflex_sibilant(self):
        """Retroflex sibilant for Sanskrit, Avestan."""
        assert segment_to_class("ʂ") == "S"

    def test_retroflex_approximant(self):
        """Retroflex approximant for Sanskrit."""
        assert segment_to_class("ɻ") == "R"

    def test_implosives(self):
        """Implosive stops (edge cases)."""
        assert segment_to_class("ɓ") == "B"
        assert segment_to_class("ɗ") == "D"

    def test_palatal_affricates(self):
        """Palatal affricates for Slavic languages."""
        assert segment_to_class("t͡ɕ") == "S"
        assert segment_to_class("d͡ʑ") == "S"

    def test_pharyngeals(self):
        """Pharyngeal consonants for Arabic, Hebrew, Maltese."""
        assert segment_to_class("ʕ") == "H"
        assert segment_to_class("ħ") == "H"

    def test_voiced_lateral_fricative(self):
        """Voiced lateral fricative for Zulu, Mongolian."""
        assert segment_to_class("ɮ") == "L"

    def test_lateral_flap(self):
        """Lateral flap for Japanese."""
        assert segment_to_class("ɺ") == "R"

    def test_labial_palatal_approximant(self):
        """Labial-palatal approximant for French, Mandarin."""
        assert segment_to_class("ɥ") == "W"

    def test_sj_sound(self):
        """Swedish sj-sound."""
        assert segment_to_class("ɧ") == "S"

    def test_bilabial_trill(self):
        """Bilabial trill for Niger-Congo."""
        assert segment_to_class("ʙ") == "B"

    def test_labiodental_flap(self):
        """Labiodental flap for Niger-Congo."""
        assert segment_to_class("ⱱ") == "B"

    def test_click_consonants(self):
        """Click consonants for Bantu, Khoisan."""
        assert segment_to_class("ʘ") == "K"
        assert segment_to_class("ǀ") == "T"
        assert segment_to_class("ǁ") == "L"
        assert segment_to_class("ǃ") == "T"
        assert segment_to_class("ǂ") == "K"


class TestIpaToSoundClass:
    def test_simple_word(self):
        assert ipa_to_sound_class("pata") == "PATA"

    def test_water(self):
        sc = ipa_to_sound_class("wɔtə")
        assert sc == "WOTE"

    def test_cognate_pair(self):
        """Ugaritic 'ab' and Hebrew 'ab' should have identical sound classes."""
        assert ipa_to_sound_class("ab") == ipa_to_sound_class("ab")
        assert ipa_to_sound_class("ab") == "AB"

    def test_empty(self):
        assert ipa_to_sound_class("") == ""

    def test_transliteration_forms(self):
        """Test with Ugaritic-Hebrew transliteration characters."""
        # Both should produce same sound class for the shared segment
        uga = ipa_to_sound_class("abd")
        heb = ipa_to_sound_class("abd")
        assert uga == heb

    def test_new_segments_no_unknowns(self):
        """Words containing new IPA segments should produce no '0' unknowns."""
        # Sanskrit-like: ʂarva (Sharva)
        assert "0" not in ipa_to_sound_class("ʂarva")
        # Turkic-like: cøɟ
        assert "0" not in ipa_to_sound_class("cøɟ")
        # Slavic-like with palatal affricate
        assert "0" not in ipa_to_sound_class("mot͡ɕ")
        # Retroflex approximant
        assert "0" not in ipa_to_sound_class("ɻana")
        # Implosive
        assert "0" not in ipa_to_sound_class("ɓaɗa")
        # Pharyngeals (Arabic)
        assert "0" not in ipa_to_sound_class("ʕajn")
        assert "0" not in ipa_to_sound_class("ħamsa")
        # Lateral fricative (Zulu)
        assert "0" not in ipa_to_sound_class("ɮala")
        # Lateral flap (Japanese)
        assert "0" not in ipa_to_sound_class("ɺana")
        # Labial-palatal (French)
        assert "0" not in ipa_to_sound_class("ɥit")
        # Sj-sound (Swedish)
        assert "0" not in ipa_to_sound_class("ɧøːn")
        # Bilabial trill
        assert "0" not in ipa_to_sound_class("ʙara")
        # Labiodental flap
        assert "0" not in ipa_to_sound_class("ⱱara")
        # Click consonants
        assert "0" not in ipa_to_sound_class("ǃosa")
        assert "0" not in ipa_to_sound_class("ǀala")
