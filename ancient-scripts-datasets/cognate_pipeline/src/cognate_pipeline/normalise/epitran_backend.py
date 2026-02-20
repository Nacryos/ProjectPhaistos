"""Epitran wrapper for grapheme-to-phoneme conversion."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cache of Epitran instances by language code
_EPITRAN_CACHE: dict[str, Any] = {}

# Mapping from Glottocode/ISO to Epitran language codes
_LANG_TO_EPITRAN: dict[str, str] = {
    "eng": "eng-Latn",
    "deu": "deu-Latn",
    "fra": "fra-Latn",
    "spa": "spa-Latn",
    "ita": "ita-Latn",
    "por": "por-Latn",
    "tur": "tur-Latn",
    "ara": "ara-Arab",
    "hin": "hin-Deva",
    "rus": "rus-Cyrl",
}


def _get_epitran(lang_code: str) -> Any | None:
    """Get or create an Epitran instance for the given language code."""
    if lang_code in _EPITRAN_CACHE:
        return _EPITRAN_CACHE[lang_code]

    try:
        import epitran
    except ImportError:
        logger.debug("epitran not installed")
        return None

    # Resolve to Epitran-format code
    epitran_code = _LANG_TO_EPITRAN.get(lang_code, lang_code)
    try:
        epi = epitran.Epitran(epitran_code)
        _EPITRAN_CACHE[lang_code] = epi
        return epi
    except Exception as exc:
        logger.debug("Epitran failed for %s: %s", epitran_code, exc)
        return None


def transliterate(text: str, lang_code: str) -> str | None:
    """Convert orthographic text to IPA using Epitran.

    Returns None if Epitran is unavailable or the language is unsupported.
    """
    epi = _get_epitran(lang_code)
    if epi is None:
        return None
    try:
        return epi.transliterate(text)
    except Exception as exc:
        logger.warning("Epitran transliteration failed for '%s' (%s): %s", text, lang_code, exc)
        return None
