"""Tests for configuration schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from cognate_pipeline.config.schema import (
    CognateConfig,
    CognateMethod,
    ClusteringAlgorithm,
    DatabaseConfig,
    NormalisationConfig,
    PipelineConfig,
    SourceDef,
    SourceFormat,
)
from cognate_pipeline.config.loader import load_config


class TestDatabaseConfig:
    def test_default_url(self):
        db = DatabaseConfig()
        assert db.url == "postgresql+psycopg://postgres:@localhost:5432/cognate_db"

    def test_custom_url(self):
        db = DatabaseConfig(host="db.example.com", port=5433, name="mydb", user="u", password="p")
        assert db.url == "postgresql+psycopg://u:p@db.example.com:5433/mydb"


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.staging_dir == Path("staging")
        assert cfg.batch_size == 5000
        assert cfg.log_level == "INFO"
        assert cfg.sources == []

    def test_normalisation_defaults(self):
        cfg = PipelineConfig()
        assert cfg.normalisation.unicode_form == "NFC"
        assert cfg.normalisation.ipa_backend_priority == ["attested", "epitran", "phonemizer"]

    def test_cognate_defaults(self):
        cfg = PipelineConfig()
        assert cfg.cognate.method == CognateMethod.BASELINE_LEV
        assert cfg.cognate.clustering == ClusteringAlgorithm.CONNECTED_COMPONENTS
        assert cfg.cognate.threshold == 0.5


class TestSourceDef:
    def test_minimal(self):
        sd = SourceDef(name="test", path=Path("data.csv"), format=SourceFormat.CSV)
        assert sd.license == "unknown"
        assert sd.encoding == "utf-8"

    def test_cog_format(self):
        sd = SourceDef(name="uga", path=Path("uga.cog"), format=SourceFormat.COG)
        assert sd.format == SourceFormat.COG


class TestLoadConfig:
    def test_load_test_config(self, config_path: Path):
        cfg = load_config(config_path)
        assert cfg.staging_dir == Path("test_staging")
        assert cfg.log_level == "DEBUG"
        assert len(cfg.sources) == 1
        assert cfg.sources[0].name == "test_cldf"
        assert cfg.sources[0].format == SourceFormat.CLDF
        assert cfg.cognate.threshold == 0.45
        assert cfg.database.name == "cognate_test_db"

    def test_load_empty_yaml(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config(p)
        assert cfg.staging_dir == Path("staging")
