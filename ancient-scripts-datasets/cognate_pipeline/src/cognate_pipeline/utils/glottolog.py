"""Glottolog tree loader and classification utilities."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Languoid:
    glottocode: str
    name: str
    iso639_3: str = ""
    family_glottocode: str = ""
    classification_path: list[str] = field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None


class GlottologTree:
    """In-memory Glottolog lookup from languoid CSV."""

    def __init__(self) -> None:
        self._by_glottocode: dict[str, Languoid] = {}
        self._by_iso: dict[str, Languoid] = {}
        self._by_name: dict[str, Languoid] = {}

    @classmethod
    def from_csv(cls, path: Path) -> GlottologTree:
        """Load from Glottolog's languoids/glottolog_languoid.csv."""
        tree = cls()
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                classification = row.get("classification", "")
                classification_path = (
                    classification.split("/") if classification else []
                )
                lang = Languoid(
                    glottocode=row["id"],
                    name=row.get("name", ""),
                    iso639_3=row.get("iso639P3code", ""),
                    family_glottocode=row.get("family_id", ""),
                    classification_path=classification_path,
                    latitude=_safe_float(row.get("latitude")),
                    longitude=_safe_float(row.get("longitude")),
                )
                tree._by_glottocode[lang.glottocode] = lang
                if lang.iso639_3:
                    tree._by_iso[lang.iso639_3] = lang
                tree._by_name[lang.name.lower()] = lang
        logger.info("Loaded %d languoids from %s", len(tree._by_glottocode), path)
        return tree

    def lookup(self, code: str) -> Languoid | None:
        """Look up by glottocode, ISO-639-3, or name (case-insensitive)."""
        if code in self._by_glottocode:
            return self._by_glottocode[code]
        if code in self._by_iso:
            return self._by_iso[code]
        return self._by_name.get(code.lower())

    def same_family(self, code_a: str, code_b: str) -> bool:
        """Check if two languoids belong to the same language family."""
        la = self.lookup(code_a)
        lb = self.lookup(code_b)
        if la is None or lb is None:
            return False
        fa = la.family_glottocode or la.glottocode
        fb = lb.family_glottocode or lb.glottocode
        return fa == fb

    def branch_distance(self, code_a: str, code_b: str) -> int | None:
        """Compute branch distance in the classification tree.

        Returns the number of steps from the lowest common ancestor
        (sum of non-shared path segments), or None if not found.
        """
        la = self.lookup(code_a)
        lb = self.lookup(code_b)
        if la is None or lb is None:
            return None
        path_a = la.classification_path
        path_b = lb.classification_path
        # Find longest common prefix
        common = 0
        for a_seg, b_seg in zip(path_a, path_b):
            if a_seg == b_seg:
                common += 1
            else:
                break
        return (len(path_a) - common) + (len(path_b) - common)


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
