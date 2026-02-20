"""Phonemizer (espeak-ng) fallback for IPA conversion."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Mapping from language codes to espeak-ng language identifiers
_LANG_TO_ESPEAK: dict[str, str] = {
    "eng": "en-us",
    "deu": "de",
    "fra": "fr-fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "tur": "tr",
}


def phonemize(text: str, lang_code: str) -> str | None:
    """Convert text to IPA using phonemizer/espeak-ng.

    Returns None if phonemizer is not installed or the language is unsupported.
    """
    try:
        from phonemizer import phonemize as _phonemize
        from phonemizer.backend import EspeakBackend
    except ImportError:
        logger.debug("phonemizer not installed")
        return None

    espeak_lang = _LANG_TO_ESPEAK.get(lang_code, lang_code)
    try:
        result = _phonemize(
            text,
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
        )
        return result if result else None
    except Exception as exc:
        logger.debug("Phonemizer failed for '%s' (%s): %s", text, lang_code, exc)
        return None
