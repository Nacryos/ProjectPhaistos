"""Unicode normalisation and IPA string cleanup."""

from __future__ import annotations

import re
import unicodedata


# Common IPA delimiters that bracket transcriptions
_IPA_BRACKET_RE = re.compile(r"^[/\[\]]+|[/\[\]]+$")

# Suprasegmental diacritics (stress marks, tone marks, length marks)
_SUPRASEGMENTAL_RE = re.compile(
    "[\u02C8\u02CC"  # Primary/secondary stress
    "\u0301\u0300\u0302\u030C\u0304\u030B\u030F"  # Tone diacritics
    "\u02E5-\u02E9"  # Tone letters
    "\u0361"  # Tie bar (could be kept for affricates)
    "]"
)

# Multiple whitespace
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalisation (NFC or NFKC)."""
    return unicodedata.normalize(form, text)


def strip_ipa_delimiters(text: str) -> str:
    """Remove enclosing /.../ or [...] brackets."""
    return _IPA_BRACKET_RE.sub("", text)


def strip_suprasegmentals(text: str) -> str:
    """Remove stress marks, tone diacritics, and length marks."""
    return _SUPRASEGMENTAL_RE.sub("", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space, strip edges."""
    return _MULTI_WS_RE.sub(" ", text).strip()


def full_cleanup(
    text: str,
    unicode_form: str = "NFC",
    strip_supra: bool = False,
    strip_ws: bool = True,
) -> str:
    """Apply all cleanup steps in sequence."""
    text = normalize_unicode(text, unicode_form)
    text = strip_ipa_delimiters(text)
    if strip_supra:
        text = strip_suprasegmentals(text)
    if strip_ws:
        text = clean_whitespace(text)
    return text
