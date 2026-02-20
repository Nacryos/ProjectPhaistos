"""Tests for candidate pair generation."""

from __future__ import annotations

from cognate_pipeline.cognate.candidate_gen import generate_candidates
from cognate_pipeline.normalise.models import NormalisedLexeme


def _make(id: str, lang: str, concept: str) -> NormalisedLexeme:
    return NormalisedLexeme(
        id=id,
        language_id=lang,
        glottocode="",
        concept_id=concept,
        form="x",
        phonetic_raw="x",
        phonetic_canonical="x",
        sound_class="X",
    )


class TestGenerateCandidates:
    def test_basic_cross_language(self):
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "heb", "water"),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_candidate"  # relationship_type

    def test_no_same_language_pairs(self):
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "uga", "water"),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 0

    def test_multiple_concepts(self):
        lexemes = [
            _make("a1", "uga", "water"),
            _make("b1", "heb", "water"),
            _make("a2", "uga", "fire"),
            _make("b2", "heb", "fire"),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 2

    def test_three_languages(self):
        """3 languages, 1 concept -> 3 pairs (C(3,2))."""
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "heb", "water"),
            _make("c", "got", "water"),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 3

    def test_empty_concept_excluded(self):
        lexemes = [
            _make("a", "uga", ""),
            _make("b", "heb", ""),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 0

    def test_single_lexeme_no_pairs(self):
        lexemes = [_make("a", "uga", "water")]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 0

    def test_ordering_consistent(self):
        lexemes = [
            _make("z_form", "uga", "water"),
            _make("a_form", "heb", "water"),
        ]
        pairs = generate_candidates(lexemes)
        assert len(pairs) == 1
        assert pairs[0][0].id < pairs[0][1].id

    def test_family_aware_semitic_pair(self):
        """Ugaritic-Hebrew pair should be tagged cognate_inherited."""
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "heb", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_cross_family(self):
        """Ugaritic-Gothic pair should be tagged similarity_only."""
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "got", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "similarity_only"

    def test_family_aware_mixed(self):
        """Three languages spanning two families."""
        lexemes = [
            _make("a", "uga", "water"),
            _make("b", "heb", "water"),
            _make("c", "got", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 3
        rel_types = {p[2] for p in pairs}
        assert "cognate_inherited" in rel_types  # uga-heb
        assert "similarity_only" in rel_types    # uga-got, heb-got

    def test_family_aware_germanic(self):
        """Gothic-Old English pair should be cognate_inherited (germanic)."""
        lexemes = [
            _make("a", "got", "water"),
            _make("b", "ang", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_celtic(self):
        """Old Irish-Welsh pair should be cognate_inherited (celtic)."""
        lexemes = [
            _make("a", "sga", "father"),
            _make("b", "cym", "father"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_balto_slavic(self):
        """Lithuanian-OCS pair should be cognate_inherited (balto_slavic)."""
        lexemes = [
            _make("a", "lit", "water"),
            _make("b", "chu", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_indo_iranian(self):
        """Sanskrit-Avestan pair should be cognate_inherited (indo_iranian)."""
        lexemes = [
            _make("a", "san", "fire"),
            _make("b", "ave", "fire"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_turkic(self):
        """Turkish-Azerbaijani pair should be cognate_inherited (turkic)."""
        lexemes = [
            _make("a", "tur", "water"),
            _make("b", "aze", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_uralic(self):
        """Finnish-Estonian pair should be cognate_inherited (uralic)."""
        lexemes = [
            _make("a", "fin", "water"),
            _make("b", "est", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "cognate_inherited"

    def test_family_aware_cross_new_families(self):
        """Germanic-Celtic pair should be similarity_only."""
        lexemes = [
            _make("a", "ang", "water"),
            _make("b", "cym", "water"),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        assert len(pairs) == 1
        assert pairs[0][2] == "similarity_only"
