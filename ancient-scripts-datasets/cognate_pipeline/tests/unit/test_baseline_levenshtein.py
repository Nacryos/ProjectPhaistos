"""Tests for baseline weighted Levenshtein scorer."""

from __future__ import annotations

from cognate_pipeline.cognate.baseline_levenshtein import (
    BaselineLevenshtein,
    normalised_similarity,
    weighted_levenshtein,
)
from cognate_pipeline.normalise.models import NormalisedLexeme


def _make_norm(
    id: str, lang: str, concept: str, form: str, sound_class: str
) -> NormalisedLexeme:
    return NormalisedLexeme(
        id=id,
        language_id=lang,
        glottocode="",
        concept_id=concept,
        form=form,
        phonetic_raw=form,
        phonetic_canonical=form,
        sound_class=sound_class,
    )


class TestWeightedLevenshtein:
    def test_identical(self):
        assert weighted_levenshtein("AB", "AB") == 0.0

    def test_empty_strings(self):
        assert weighted_levenshtein("", "") == 0.0

    def test_one_empty(self):
        assert weighted_levenshtein("AB", "") == 1.0  # 2 * 0.5
        assert weighted_levenshtein("", "ABC") == 1.5

    def test_same_class_cheaper(self):
        """Substitution within same class should be cheaper than across classes."""
        cost_same_class = weighted_levenshtein("P", "B")
        cost_diff_class = weighted_levenshtein("P", "T")
        assert cost_same_class < cost_diff_class

    def test_vowel_substitution_cheap(self):
        cost = weighted_levenshtein("A", "E")
        assert cost == 0.3  # Both vowels


class TestNormalisedSimilarity:
    def test_identical_is_one(self):
        assert normalised_similarity("AB", "AB") == 1.0

    def test_empty_is_one(self):
        assert normalised_similarity("", "") == 1.0

    def test_range_zero_to_one(self):
        sim = normalised_similarity("AB", "KG")
        assert 0.0 <= sim <= 1.0


class TestBaselineLevenshtein:
    def test_cognate_pair_above_threshold(self):
        """Identical sound classes should score 1.0."""
        a = _make_norm("uga_1", "uga", "father", "ab", "AB")
        b = _make_norm("heb_1", "heb", "father", "ab", "AB")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs([(a, b, "cognate_inherited")], threshold=0.5)
        assert len(links) == 1
        assert links[0].score == 1.0
        assert links[0].relationship_type == "cognate_inherited"

    def test_below_threshold_filtered(self):
        a = _make_norm("a_1", "x", "c1", "xxx", "KKK")
        b = _make_norm("b_1", "y", "c1", "yyy", "MNR")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs([(a, b, "cognate_candidate")], threshold=0.9)
        assert len(links) == 0

    def test_evidence_present(self):
        a = _make_norm("a_1", "x", "c1", "pa", "PA")
        b = _make_norm("b_1", "y", "c1", "ba", "BA")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs([(a, b, "cognate_candidate")], threshold=0.0)
        assert len(links) == 1
        assert "sound_class_a" in links[0].evidence
        assert "distance" in links[0].evidence

    def test_ordering_consistency(self):
        a = _make_norm("z_1", "x", "c1", "a", "A")
        b = _make_norm("a_1", "y", "c1", "a", "A")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs([(a, b, "cognate_candidate")], threshold=0.0)
        assert links[0].lexeme_id_a < links[0].lexeme_id_b

    def test_relationship_type_preserved(self):
        """The relationship type from candidate_gen should be preserved."""
        a = _make_norm("uga_1", "uga", "c1", "ab", "AB")
        b = _make_norm("got_1", "got", "c1", "ab", "AB")
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs([(a, b, "similarity_only")], threshold=0.0)
        assert links[0].relationship_type == "similarity_only"
