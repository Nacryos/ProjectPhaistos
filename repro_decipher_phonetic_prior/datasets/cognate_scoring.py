"""Cognate confidence scoring for validation branch ground-truth pairs.

Provides a pure-Python cognate scoring pipeline using:
  1. Normalized Levenshtein edit distance on raw forms
  2. SCA (Sound-Class-based Alignment) encoding with class-aware distance
  3. Combined confidence score

No C extensions required — all computations are in pure Python.

Usage::

    from datasets.cognate_scoring import score_cognate_pair, annotate_ground_truth

    conf = score_cognate_pair("hund", "hound")
    gt = annotate_ground_truth(corpus.ground_truth, corpus.known_text)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# SCA sound-class encoding (Dolgopolsky-style, extended)
# ---------------------------------------------------------------------------
# Maps IPA characters to broad sound classes.  Characters in the same class
# are treated as "similar" for cognate detection.  Based on Dolgopolsky (1964)
# and List (2012) SCA scheme, simplified for the character inventories we
# encounter in NorthEuraLex / WOLD / ancient-scripts data.

_SCA_MAP: Dict[str, str] = {}

# Build mapping: each class letter maps a set of IPA chars to itself.
_SCA_CLASSES = {
    "P": "pbɸβ",           # labial stops / fricatives
    "T": "tdtθðʈɖ",        # dental/alveolar stops
    "K": "kgqɢɡ",          # velar/uvular stops
    "S": "szʃʒɕʑʂʐçɧʝ",   # sibilants / fricatives
    "C": "tsʦdzʣtʃʧdʒʤ",  # affricates
    "M": "mnɱɲŋɴ",         # nasals
    "N": "nɲŋɴ",           # (merged with M for scoring, but distinct for alignment)
    "L": "lɫɭʎɺɬɮ",       # laterals
    "R": "rɾɹʀɽʁ",        # rhotics
    "W": "wʋɥ",            # labial glides
    "J": "jʝ",             # palatal glides
    "H": "hɦħʕʔʜʢ",       # laryngeals / pharyngeals
    "V": "aeiouæøəɐɑɒɛɜɤɔɪʊʏɵɶœɘ",  # vowels
    "F": "fvxɣχʁ",         # non-sibilant fricatives
}

for cls_char, ipa_chars in _SCA_CLASSES.items():
    for ch in ipa_chars:
        _SCA_MAP[ch] = cls_char

# Common diacritics / modifiers to strip before lookup.
_DIACRITIC_STRIP = "ːˑʰʷʲˠˤ̩̥̤̰̃̊̍̆̂̄ʼ"


def _sca_encode(token: str) -> str:
    """Encode a token into its SCA sound-class string."""
    out: list[str] = []
    for ch in token:
        if ch in _DIACRITIC_STRIP or ch == " ":
            continue
        cls = _SCA_MAP.get(ch)
        if cls:
            # Collapse consecutive identical classes.
            if not out or out[-1] != cls:
                out.append(cls)
        # Unknown chars: skip (punctuation, digits, etc.)
    return "".join(out)


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python)
# ---------------------------------------------------------------------------

def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use single-row DP for memory efficiency.
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _normalized_edit_distance(s: str, t: str) -> float:
    """Normalized edit distance in [0, 1].  0 = identical, 1 = maximally different."""
    if not s and not t:
        return 0.0
    dist = _levenshtein(s, t)
    return dist / max(len(s), len(t))


# ---------------------------------------------------------------------------
# SCA-aware distance
# ---------------------------------------------------------------------------

# Substitution costs within/across SCA classes.
_SCA_SIMILAR_GROUPS = [
    {"P", "F"},     # labial stops ~ labial fricatives
    {"T", "S"},     # dental stops ~ sibilants
    {"K", "F"},     # velar stops ~ velar fricatives
    {"M", "N"},     # nasals
    {"L", "R"},     # liquids
    {"W", "J"},     # glides
]


def _sca_substitution_cost(a: str, b: str) -> float:
    """Substitution cost between two SCA classes."""
    if a == b:
        return 0.0
    for group in _SCA_SIMILAR_GROUPS:
        if a in group and b in group:
            return 0.5
    # Vowel vs vowel is free (vowels shift freely in cognates).
    if a == "V" and b == "V":
        return 0.0
    return 1.0


def _sca_distance(s: str, t: str) -> float:
    """Weighted edit distance on SCA-encoded strings."""
    m, n = len(s), len(t)
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)

    prev = [float(j) for j in range(n + 1)]
    curr = [0.0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = float(i)
        for j in range(1, n + 1):
            cost = _sca_substitution_cost(s[i - 1], t[j - 1])
            curr[j] = min(
                prev[j] + 1.0,        # deletion
                curr[j - 1] + 1.0,    # insertion
                prev[j - 1] + cost     # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _normalized_sca_distance(s: str, t: str) -> float:
    """Normalized SCA distance in [0, 1]."""
    if not s and not t:
        return 0.0
    dist = _sca_distance(s, t)
    return dist / max(len(s), len(t))


# ---------------------------------------------------------------------------
# Combined cognate confidence score
# ---------------------------------------------------------------------------

def score_cognate_pair(
    lost_form: str,
    known_form: str,
    *,
    raw_weight: float = 0.3,
    sca_weight: float = 0.7,
) -> float:
    """Score a (lost, known) pair for cognate likelihood.

    Returns a confidence in [0, 1] where 1 = very likely cognate.
    Combines raw normalized edit distance (30%) with SCA-encoded
    distance (70%) — SCA is more informative because it groups
    similar sounds.

    Parameters
    ----------
    lost_form : str
        The lost-language surface form (IPA or transliteration).
    known_form : str
        The known-language surface form.
    raw_weight : float
        Weight for raw character edit distance component.
    sca_weight : float
        Weight for SCA-encoded distance component.
    """
    raw_dist = _normalized_edit_distance(lost_form, known_form)
    sca_lost = _sca_encode(lost_form)
    sca_known = _sca_encode(known_form)
    sca_dist = _normalized_sca_distance(sca_lost, sca_known)

    combined = raw_weight * raw_dist + sca_weight * sca_dist
    return max(0.0, min(1.0, 1.0 - combined))


def annotate_ground_truth(
    ground_truth: Sequence[Dict[str, Any]],
    known_text: Optional[Dict[str, List[str]]] = None,
    *,
    raw_weight: float = 0.3,
    sca_weight: float = 0.7,
) -> List[Dict[str, Any]]:
    """Add ``cognate_score`` to each ground-truth row.

    Each row must have ``lost`` (str) and ``known`` (list of str or str).
    Returns a new list of dicts with the added ``cognate_score`` field
    (average score across all known candidates for that row) and
    ``cognate_scores`` (per-candidate scores).

    Parameters
    ----------
    ground_truth : sequence of dicts
        Ground-truth rows from a Corpus.
    known_text : dict, optional
        Not used directly; reserved for future vocabulary-based scoring.
    raw_weight : float
        Weight for raw edit distance in the scoring function.
    sca_weight : float
        Weight for SCA distance in the scoring function.
    """
    annotated: List[Dict[str, Any]] = []
    for row in ground_truth:
        new_row = dict(row)
        lost = row.get("lost", "")
        known_candidates = row.get("known", [])
        if isinstance(known_candidates, str):
            known_candidates = [known_candidates] if known_candidates else []

        scores: List[float] = []
        for known in known_candidates:
            if lost and known:
                scores.append(
                    score_cognate_pair(lost, known, raw_weight=raw_weight, sca_weight=sca_weight)
                )

        if scores:
            new_row["cognate_score"] = sum(scores) / len(scores)
            new_row["cognate_scores"] = {k: s for k, s in zip(known_candidates, scores)}
        else:
            new_row["cognate_score"] = 0.0
            new_row["cognate_scores"] = {}

        annotated.append(new_row)
    return annotated


def sca_encode(token: str) -> str:
    """Public API: encode a token into SCA sound classes."""
    return _sca_encode(token)
