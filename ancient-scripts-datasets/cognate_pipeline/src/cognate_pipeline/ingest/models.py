"""Data models for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognate_pipeline.provenance.tracker import ProvenanceRecord


class TranscriptionType:
    """Constants for the type of phonetic representation stored."""

    IPA = "ipa"                          # True IPA transcription
    TRANSLITERATION = "transliteration"  # Script transliteration (e.g. Ugaritic consonantal)
    ORTHOGRAPHIC = "orthographic"        # Standard orthography (e.g. Gothic Latin letters)
    UNKNOWN = "unknown"


@dataclass
class RawLexeme:
    """A single lexical form as ingested from a source."""

    id: str
    language_id: str
    glottocode: str
    concept_id: str
    form: str
    phonetic_raw: str = ""
    transcription_type: str = TranscriptionType.UNKNOWN
    alternatives: list[str] = field(default_factory=list)
    source_name: str = ""
    provenance: ProvenanceRecord | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible alias
    @property
    def ipa_raw(self) -> str:
        return self.phonetic_raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language_id": self.language_id,
            "glottocode": self.glottocode,
            "concept_id": self.concept_id,
            "form": self.form,
            "phonetic_raw": self.phonetic_raw,
            "transcription_type": self.transcription_type,
            "alternatives": self.alternatives,
            "source_name": self.source_name,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawLexeme:
        prov = d.get("provenance")
        return cls(
            id=d["id"],
            language_id=d["language_id"],
            glottocode=d.get("glottocode", ""),
            concept_id=d.get("concept_id", ""),
            form=d["form"],
            phonetic_raw=d.get("phonetic_raw", d.get("ipa_raw", "")),
            transcription_type=d.get("transcription_type", TranscriptionType.UNKNOWN),
            alternatives=d.get("alternatives", []),
            source_name=d.get("source_name", ""),
            provenance=ProvenanceRecord.from_dict(prov) if prov else None,
            extra=d.get("extra", {}),
        )


@dataclass
class RawNameForm:
    """A named entity form (place name, personal name, etc.)."""

    id: str
    entity_type: str
    language_id: str
    glottocode: str
    name_string: str
    ipa_raw: str = ""
    source_name: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)
    latitude: float | None = None
    longitude: float | None = None
    provenance: ProvenanceRecord | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "language_id": self.language_id,
            "glottocode": self.glottocode,
            "name_string": self.name_string,
            "ipa_raw": self.ipa_raw,
            "source_name": self.source_name,
            "external_ids": self.external_ids,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawNameForm:
        prov = d.get("provenance")
        return cls(
            id=d["id"],
            entity_type=d["entity_type"],
            language_id=d["language_id"],
            glottocode=d.get("glottocode", ""),
            name_string=d["name_string"],
            ipa_raw=d.get("ipa_raw", ""),
            source_name=d.get("source_name", ""),
            external_ids=d.get("external_ids", {}),
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            provenance=ProvenanceRecord.from_dict(prov) if prov else None,
            extra=d.get("extra", {}),
        )
