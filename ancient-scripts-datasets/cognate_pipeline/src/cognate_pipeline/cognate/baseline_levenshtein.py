"""Weighted Levenshtein scorer using SCA sound classes."""

from __future__ import annotations

import logging
from cognate_pipeline.cognate.models import CognateLink
from cognate_pipeline.normalise.models import NormalisedLexeme

logger = logging.getLogger(__name__)

# Substitution costs between SCA classes (lower = more similar)
# Same class = 0, vowels among themselves = 0.3, related consonants = 0.5, unrelated = 1.0
_VOWELS = set("AEIOU")
_LABIALS = {"P", "B", "M"}
_CORONALS = {"T", "D", "N", "S", "L", "R"}
_VELARS = {"K", "G"}
_LARYNGEALS = {"H"}
_GLIDES = {"W", "Y"}

_NATURAL_CLASSES = [_VOWELS, _LABIALS, _CORONALS, _VELARS, _LARYNGEALS, _GLIDES]


def _substitution_cost(a: str, b: str) -> float:
    """Compute substitution cost between two SCA class characters."""
    if a == b:
        return 0.0
    # Check if in same natural class
    for cls in _NATURAL_CLASSES:
        if a in cls and b in cls:
            return 0.3
    return 1.0


def weighted_levenshtein(s1: str, s2: str) -> float:
    """Compute weighted Levenshtein distance using SCA-aware costs.

    Insertion/deletion cost = 0.5, substitution cost varies by class.
    """
    n, m = len(s1), len(s2)
    if n == 0:
        return m * 0.5
    if m == 0:
        return n * 0.5

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * 0.5
    for j in range(m + 1):
        dp[0][j] = j * 0.5

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = _substitution_cost(s1[i - 1], s2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 0.5,      # deletion
                dp[i][j - 1] + 0.5,       # insertion
                dp[i - 1][j - 1] + sub_cost,  # substitution
            )
    return dp[n][m]


def normalised_similarity(s1: str, s2: str) -> float:
    """Compute normalised similarity (0-1) from weighted Levenshtein.

    1.0 = identical, 0.0 = maximally different.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    # Maximum possible distance is max_len * 1.0 (all substitutions at full cost)
    dist = weighted_levenshtein(s1, s2)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


class BaselineLevenshtein:
    """Baseline cognate detector using weighted Levenshtein on sound classes."""

    def score_pairs(
        self,
        pairs: list[tuple[NormalisedLexeme, NormalisedLexeme, str]],
        threshold: float = 0.5,
    ) -> list[CognateLink]:
        """Score all candidate pairs and return links above threshold.

        Each pair is (lexeme_a, lexeme_b, relationship_type).
        """
        links: list[CognateLink] = []
        for a, b, rel_type in pairs:
            sc_a = a.sound_class
            sc_b = b.sound_class
            score = normalised_similarity(sc_a, sc_b)
            if score >= threshold:
                # Ensure consistent ordering
                id_a, id_b = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                links.append(
                    CognateLink(
                        lexeme_id_a=id_a,
                        lexeme_id_b=id_b,
                        concept_id=a.concept_id,
                        relationship_type=rel_type,
                        score=round(score, 4),
                        method="baseline_lev",
                        threshold_used=threshold,
                        evidence={
                            "sound_class_a": sc_a,
                            "sound_class_b": sc_b,
                            "distance": round(weighted_levenshtein(sc_a, sc_b), 4),
                        },
                    )
                )
        logger.info("Scored %d pairs, %d above threshold %.2f", len(pairs), len(links), threshold)
        return links
