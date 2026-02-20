"""Sound Correspondence Analysis (SCA) tokeniser and encoder.

Maps IPA segments to single-character sound classes based on the SCA
system (List 2012). This allows efficient comparison of phonological
patterns across languages.
"""

from __future__ import annotations

import re

# SCA sound class mapping: IPA segment -> class character
# Based on List (2012) "SCA: A Method for Automatic Sound Correspondence Analysis"
_SCA_MAP: dict[str, str] = {
    # Vowels -> V
    "a": "A", "e": "E", "i": "I", "o": "O", "u": "U",
    "ɑ": "A", "æ": "A", "ɐ": "A",
    "ɛ": "E", "ə": "E", "ɘ": "E",
    "ɪ": "I", "ɨ": "I",
    "ɔ": "O", "ɵ": "O",
    "ʊ": "U", "ʉ": "U", "ɯ": "U",
    "y": "U", "ø": "E", "œ": "E",

    # Labial stops
    "p": "P", "b": "B", "ɸ": "P", "β": "B",

    # Alveolar stops
    "t": "T", "d": "D", "ʈ": "T", "ɖ": "D",

    # Velar stops
    "k": "K", "g": "G", "ɡ": "G", "q": "K", "ɢ": "G",

    # Glottal / Pharyngeal
    "ʔ": "H", "h": "H", "ɦ": "H",
    "ʕ": "H", "ħ": "H",  # pharyngeals (Arabic, Hebrew, Maltese)

    # Nasals
    "m": "M", "n": "N", "ɲ": "N", "ŋ": "N", "ɳ": "N", "ɴ": "N",

    # Liquids
    "l": "L", "ɫ": "L", "ɭ": "L", "ɬ": "L",
    "ɮ": "L",  # voiced lateral fricative (Zulu, Mongolian)
    "r": "R", "ɾ": "R", "ɽ": "R", "ʀ": "R", "ɹ": "R", "ʁ": "R",
    "ɺ": "R",  # lateral flap (Japanese)

    # Fricatives
    "f": "P", "v": "B",
    "s": "S", "z": "S", "ʃ": "S", "ʒ": "S",
    "ɕ": "S", "ʑ": "S",
    "θ": "T", "ð": "D",
    "x": "K", "ɣ": "G", "χ": "K", "ʝ": "G",

    # Palatal stops (Turkic, Celtic, Hungarian)
    "c": "K", "ɟ": "G",

    # Retroflex sibilant and approximant (Sanskrit, Avestan)
    "ʂ": "S", "ɻ": "R",
    "ɧ": "S",  # sj-sound (Swedish)

    # Click consonants (Bantu, Khoisan)
    "ʘ": "K",  # bilabial click
    "ǀ": "T",  # dental click
    "ǁ": "L",  # lateral click
    "ǃ": "T",  # alveolar click
    "ǂ": "K",  # palatal click

    # Implosives / trills / flaps
    "ɓ": "B", "ɗ": "D",
    "ʙ": "B",  # bilabial trill (Niger-Congo)
    "ⱱ": "B",  # labiodental flap (Niger-Congo)

    # Affricates (common)
    "t͡s": "S", "d͡z": "S", "t͡ʃ": "S", "d͡ʒ": "S",
    "t͡ɕ": "S", "d͡ʑ": "S",  # palatal affricates (Slavic)
    "ts": "S", "dz": "S", "tʃ": "S", "dʒ": "S",
    "tɕ": "S", "dʑ": "S",  # palatal affricates (no tie bar)

    # Glides
    "w": "W", "j": "Y", "ʋ": "W", "ɰ": "W",
    "ɥ": "W",  # labial-palatal approximant (French, Mandarin)

    # Transliteration characters (for ancient scripts)
    # These map based on their phonological values
    "$": "S",  # Shin/Sibilant
    "H": "H",  # Het/Pharyngeal
    "<": "H",  # Ayin
    "@": "S",  # Tsade
    "*": "S",  # Emphatic sibilant
}

# Pattern to split IPA into segments (handles affricates with tie bar)
_SEGMENT_RE = re.compile(
    r"[a-zA-Zɑæɐɛəɘɪɨɔɵʊʉɯøœɸβʈɖɡɢʔɦɲŋɳɴɫɭɬɮɾɽʀɹʁɕʑʃʒθðɣχʝʋɰɟʂɻɓɗʕħɺɥɧʙⱱʘǀǁǃǂ$H<@*]"
    r"(?:\u0361[a-zA-Zɑæɐɛəɘɪɨɔɵʊʉɯøœɸβʈɖɡɢʔɦɲŋɳɴɫɭɬɮɾɽʀɹʁɕʑʃʒθðɣχʝʋɰɟʂɻɓɗʕħɺɥɧʙⱱʘǀǁǃǂ])?"
    r"[\u0300-\u036F\u0325\u0329\u032A\u033A\u033B\u033C\u02B0\u02BC\u02D0\u02D1]*"
)


def tokenize_ipa(ipa: str) -> list[str]:
    """Split an IPA string into a list of segments.

    Handles diacritics by attaching them to the preceding base character,
    and recognizes affricates written with tie bars.
    """
    if not ipa:
        return []
    return _SEGMENT_RE.findall(ipa)


def segment_to_class(segment: str) -> str:
    """Map a single IPA segment to its SCA class character."""
    # Strip diacritics for lookup
    base = segment[0] if segment else ""
    # Check multi-char first (affricates)
    if len(segment) >= 2:
        digraph = segment[:2]
        if digraph in _SCA_MAP:
            return _SCA_MAP[digraph]
        # Check with tie bar
        if len(segment) >= 3 and segment[1] == "\u0361":
            affricate = segment[0] + segment[2]
            if affricate in _SCA_MAP:
                return _SCA_MAP[affricate]
    # Single char lookup
    if base in _SCA_MAP:
        return _SCA_MAP[base]
    return "0"  # Unknown


def ipa_to_sound_class(ipa: str) -> str:
    """Convert an IPA string to a compact SCA sound-class string."""
    segments = tokenize_ipa(ipa)
    return "".join(segment_to_class(s) for s in segments)
