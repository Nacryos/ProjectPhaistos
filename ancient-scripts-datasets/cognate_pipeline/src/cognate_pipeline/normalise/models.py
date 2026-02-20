"""Data models for the normalisation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognate_pipeline.provenance.tracker import ProvenanceRecord


@dataclass
class NormalisedLexeme:
    """A lexeme after phonetic normalisation and sound-class encoding.

    Fields distinguish three representations:
    - ``phonetic_raw``: the raw phonetic string from the source (may be IPA,
      transliteration, or orthographic — ``transcription_type`` says which)
    - ``phonetic_canonical``: the cleaned/normalised form used for comparison
    - ``sound_class``: the SCA encoding derived from ``phonetic_canonical``
    """

    id: str
    language_id: str
    glottocode: str
    concept_id: str
    form: str
    phonetic_raw: str
    phonetic_canonical: str
    sound_class: str
    transcription_type: str = "unknown"
    confidence: float = 1.0
    alternatives: list[str] = field(default_factory=list)
    source_name: str = ""
    provenance: ProvenanceRecord | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible aliases
    @property
    def ipa_raw(self) -> str:
        return self.phonetic_raw

    @property
    def ipa_canonical(self) -> str:
        return self.phonetic_canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language_id": self.language_id,
            "glottocode": self.glottocode,
            "concept_id": self.concept_id,
            "form": self.form,
            "phonetic_raw": self.phonetic_raw,
            "phonetic_canonical": self.phonetic_canonical,
            "sound_class": self.sound_class,
            "transcription_type": self.transcription_type,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "source_name": self.source_name,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalisedLexeme:
        prov = d.get("provenance")
        return cls(
            id=d["id"],
            language_id=d["language_id"],
            glottocode=d.get("glottocode", ""),
            concept_id=d.get("concept_id", ""),
            form=d["form"],
            phonetic_raw=d.get("phonetic_raw", d.get("ipa_raw", "")),
            phonetic_canonical=d.get("phonetic_canonical", d.get("ipa_canonical", "")),
            sound_class=d.get("sound_class", ""),
            transcription_type=d.get("transcription_type", "unknown"),
            confidence=d.get("confidence", 1.0),
            alternatives=d.get("alternatives", []),
            source_name=d.get("source_name", ""),
            provenance=ProvenanceRecord.from_dict(prov) if prov else None,
            extra=d.get("extra", {}),
        )
