"""Common project paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
DATA = ROOT / "data"
DATA_PREPARED = DATA / "prepared"
DATA_EXTERNAL = ROOT / "data_external"
SCRIPTS = ROOT / "scripts"
PATCHES = ROOT / "patches"


def ensure_layout() -> None:
    for path in (ARTIFACTS, FIGURES, DATA, DATA_PREPARED, DATA_EXTERNAL, PATCHES):
        path.mkdir(parents=True, exist_ok=True)
