"""Data models for cognate detection and clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CognateLink:
    """A scored pairwise cognate relationship between two lexemes."""

    lexeme_id_a: str
    lexeme_id_b: str
    concept_id: str
    relationship_type: str = "cognate_candidate"  # cognate_inherited | similarity_only | cognate_candidate
    score: float = 0.0
    method: str = ""
    threshold_used: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexeme_id_a": self.lexeme_id_a,
            "lexeme_id_b": self.lexeme_id_b,
            "concept_id": self.concept_id,
            "relationship_type": self.relationship_type,
            "score": self.score,
            "method": self.method,
            "threshold_used": self.threshold_used,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CognateLink:
        return cls(
            lexeme_id_a=d["lexeme_id_a"],
            lexeme_id_b=d["lexeme_id_b"],
            concept_id=d.get("concept_id", ""),
            relationship_type=d.get("relationship_type", "cognate_candidate"),
            score=d.get("score", 0.0),
            method=d.get("method", ""),
            threshold_used=d.get("threshold_used", 0.0),
            evidence=d.get("evidence", {}),
        )


@dataclass
class CognateSetMember:
    """A member of a cognate set."""

    lexeme_id: str
    role: str = "member"  # member | proto | reflex

    def to_dict(self) -> dict[str, Any]:
        return {"lexeme_id": self.lexeme_id, "role": self.role}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CognateSetMember:
        return cls(lexeme_id=d["lexeme_id"], role=d.get("role", "member"))


@dataclass
class CognateSet:
    """A cluster of cognate lexemes sharing a common etymon."""

    id: str
    concept_id: str
    method: str
    members: list[CognateSetMember] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "method": self.method,
            "members": [m.to_dict() for m in self.members],
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CognateSet:
        return cls(
            id=d["id"],
            concept_id=d.get("concept_id", ""),
            method=d.get("method", ""),
            members=[CognateSetMember.from_dict(m) for m in d.get("members", [])],
            quality=d.get("quality", {}),
        )
