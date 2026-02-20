"""Load and validate pipeline configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PipelineConfig


def load_config(path: Path | str) -> PipelineConfig:
    """Read a YAML file and return a validated PipelineConfig."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    return PipelineConfig.model_validate(raw)
