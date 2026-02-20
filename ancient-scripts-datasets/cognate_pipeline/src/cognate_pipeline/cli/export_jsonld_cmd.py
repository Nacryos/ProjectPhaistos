"""CLI handler for the export-jsonld subcommand."""

from __future__ import annotations

import logging

from cognate_pipeline.config.loader import load_config
from cognate_pipeline.db.connection import get_engine
from cognate_pipeline.export.jsonld_exporter import JsonLdExporter
from cognate_pipeline.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_export_jsonld(config_path: str) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)

    engine = get_engine(cfg.database)
    exporter = JsonLdExporter(engine, cfg.export)
    out_dir = cfg.export.jsonld_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter.export(out_dir)
    logger.info("JSON-LD export written to %s", out_dir)
