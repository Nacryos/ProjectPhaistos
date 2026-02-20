"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def config_path() -> Path:
    return FIXTURES_DIR / "config_test.yaml"


@pytest.fixture
def sample_cldf_dir() -> Path:
    return FIXTURES_DIR / "sample_cldf"


@pytest.fixture
def tmp_staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    return staging
