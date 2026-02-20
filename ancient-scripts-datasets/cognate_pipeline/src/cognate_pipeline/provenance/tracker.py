"""Provenance tracking for pipeline transformations."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProvenanceStep:
    """A single transformation step in the provenance chain."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "params": self.params,
            "result": self.result,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceStep:
        return cls(
            tool=d["tool"],
            params=d.get("params", {}),
            result=d.get("result", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class ProvenanceRecord:
    """Full provenance chain for a data item."""

    source_name: str
    source_format: str
    original_id: str = ""
    steps: list[ProvenanceStep] = field(default_factory=list)

    def add_step(
        self, tool: str, params: dict[str, Any] | None = None, result: str = ""
    ) -> ProvenanceRecord:
        self.steps.append(
            ProvenanceStep(tool=tool, params=params or {}, result=result)
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_format": self.source_format,
            "original_id": self.original_id,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            source_name=d["source_name"],
            source_format=d["source_format"],
            original_id=d.get("original_id", ""),
            steps=[ProvenanceStep.from_dict(s) for s in d.get("steps", [])],
        )
