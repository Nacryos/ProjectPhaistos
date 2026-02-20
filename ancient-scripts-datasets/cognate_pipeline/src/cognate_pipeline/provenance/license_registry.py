"""Per-source license and citation metadata store."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LicenseEntry:
    source_name: str
    license: str
    license_url: str = ""
    citation_bibtex: str = ""
    retrieved_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "license": self.license,
            "license_url": self.license_url,
            "citation_bibtex": self.citation_bibtex,
            "retrieved_at": self.retrieved_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LicenseEntry:
        return cls(**d)


class LicenseRegistry:
    """Collects license metadata for all ingested sources."""

    def __init__(self) -> None:
        self._entries: dict[str, LicenseEntry] = {}

    def register(
        self,
        source_name: str,
        license_: str,
        license_url: str = "",
        citation_bibtex: str = "",
    ) -> None:
        self._entries[source_name] = LicenseEntry(
            source_name=source_name,
            license=license_,
            license_url=license_url,
            citation_bibtex=citation_bibtex,
        )

    def get(self, source_name: str) -> LicenseEntry | None:
        return self._entries.get(source_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": {k: v.to_dict() for k, v in self._entries.items()}
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LicenseRegistry:
        reg = cls()
        for entry_data in d.get("sources", {}).values():
            entry = LicenseEntry.from_dict(entry_data)
            reg._entries[entry.source_name] = entry
        return reg
