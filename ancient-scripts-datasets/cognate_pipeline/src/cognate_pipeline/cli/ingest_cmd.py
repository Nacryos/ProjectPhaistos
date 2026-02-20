"""CLI handler for the ingest-sources subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

import orjson

from cognate_pipeline.config.loader import load_config
from cognate_pipeline.config.schema import SourceFormat
from cognate_pipeline.ingest.base import SourceIngester
from cognate_pipeline.ingest.csv_ingester import CsvIngester
from cognate_pipeline.ingest.cldf_ingester import CldfIngester
from cognate_pipeline.ingest.json_ingester import JsonIngester
from cognate_pipeline.ingest.wiktionary_ingester import WiktionaryIngester
from cognate_pipeline.provenance.license_registry import LicenseRegistry
from cognate_pipeline.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

_FORMAT_TO_INGESTER: dict[SourceFormat, type[SourceIngester]] = {
    SourceFormat.CLDF: CldfIngester,
    SourceFormat.CSV: CsvIngester,
    SourceFormat.TSV: CsvIngester,
    SourceFormat.COG: CsvIngester,
    SourceFormat.JSON: JsonIngester,
    SourceFormat.NDJSON: JsonIngester,
    SourceFormat.WIKTIONARY: WiktionaryIngester,
}


def run_ingest(config_path: str, output_dir: str | None) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)
    registry = LicenseRegistry()

    staging = Path(output_dir) if output_dir else cfg.staging_dir / "ingest"
    staging.mkdir(parents=True, exist_ok=True)

    for source_def in cfg.sources:
        logger.info("Ingesting source: %s (%s)", source_def.name, source_def.format)
        registry.register(
            source_def.name, source_def.license, source_def.license_url,
            source_def.citation_bibtex,
        )
        ingester_cls = _FORMAT_TO_INGESTER.get(source_def.format)
        if ingester_cls is None:
            logger.error("No ingester for format %s", source_def.format)
            continue

        ingester = ingester_cls(source_def)
        out_path = staging / f"{source_def.name}.jsonl"
        count = 0
        with out_path.open("wb") as fh:
            for lexeme in ingester.ingest():
                fh.write(orjson.dumps(lexeme.to_dict()) + b"\n")
                count += 1
        logger.info("  Wrote %d lexemes to %s", count, out_path)

    reg_path = staging / "_license_registry.json"
    reg_path.write_bytes(orjson.dumps(registry.to_dict()))
    logger.info("License registry saved to %s", reg_path)
