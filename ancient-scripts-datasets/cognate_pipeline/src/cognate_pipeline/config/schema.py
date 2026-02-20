"""Pydantic v2 configuration models for the cognate pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CognateMethod(str, Enum):
    BASELINE_LEV = "baseline_lev"
    LEXSTAT = "lexstat"


class ClusteringAlgorithm(str, Enum):
    UPGMA = "upgma"
    CONNECTED_COMPONENTS = "connected_components"


class SourceFormat(str, Enum):
    CLDF = "cldf"
    CSV = "csv"
    TSV = "tsv"
    COG = "cog"
    JSON = "json"
    NDJSON = "ndjson"
    WIKTIONARY = "wiktionary"


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "cognate_db"
    user: str = "postgres"
    password: str = ""
    schema_name: str = "public"

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class ColumnMapping(BaseModel):
    """Maps source columns to internal fields."""

    language: str = "Language_ID"
    form: str = "Form"
    concept: str = "Parameter_ID"
    ipa: str | None = None
    glottocode: str | None = None
    source_id: str | None = None


class SourceDef(BaseModel):
    """Definition of a single data source."""

    name: str
    path: Path
    format: SourceFormat
    license: str = "unknown"
    license_url: str = ""
    citation_bibtex: str = ""
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    delimiter: str | None = None
    encoding: str = "utf-8"
    extra: dict[str, Any] = Field(default_factory=dict)


class NormalisationConfig(BaseModel):
    unicode_form: str = "NFC"
    strip_suprasegmentals: bool = False
    strip_whitespace: bool = True
    ipa_backend_priority: list[str] = Field(
        default_factory=lambda: ["attested", "epitran", "phonemizer"]
    )
    transliteration_passthrough: bool = True


class CognateConfig(BaseModel):
    method: CognateMethod = CognateMethod.BASELINE_LEV
    clustering: ClusteringAlgorithm = ClusteringAlgorithm.CONNECTED_COMPONENTS
    threshold: float = 0.5
    family_aware: bool = True
    batch_size: int = 10000


class ExportConfig(BaseModel):
    cldf_output_dir: Path = Path("export/cldf")
    jsonld_output_dir: Path = Path("export/jsonld")
    include_provenance: bool = True


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    staging_dir: Path = Path("staging")
    sources: list[SourceDef] = Field(default_factory=list)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    normalisation: NormalisationConfig = Field(default_factory=NormalisationConfig)
    cognate: CognateConfig = Field(default_factory=CognateConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    glottolog_data_dir: Path | None = None
    batch_size: int = 5000
    log_level: str = "INFO"
