"""CLI handler for the load-db subcommand."""

from __future__ import annotations

import logging

import orjson

from cognate_pipeline.config.loader import load_config
from cognate_pipeline.db.connection import get_engine, get_session
from cognate_pipeline.db.loader import BatchLoader
from cognate_pipeline.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_load(config_path: str) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)

    engine = get_engine(cfg.database)
    loader = BatchLoader(engine, batch_size=cfg.batch_size)

    # Load license registry
    reg_path = cfg.staging_dir / "ingest" / "_license_registry.json"
    if reg_path.exists():
        registry = orjson.loads(reg_path.read_bytes())
        loader.load_sources(registry)
        logger.info("Loaded source metadata")

    # Load normalised lexemes
    norm_dir = cfg.staging_dir / "normalised"
    for jsonl_path in sorted(norm_dir.glob("*.jsonl")):
        if jsonl_path.name.startswith("_"):
            continue
        logger.info("Loading lexemes from %s", jsonl_path.name)
        records = []
        with jsonl_path.open("rb") as fh:
            for line in fh:
                records.append(orjson.loads(line))
        loader.load_lexemes(records)

    # Load cognate data
    cog_dir = cfg.staging_dir / "cognate"
    links_path = cog_dir / "cognate_links.jsonl"
    if links_path.exists():
        links = []
        with links_path.open("rb") as fh:
            for line in fh:
                links.append(orjson.loads(line))
        loader.load_cognate_links(links)
        logger.info("Loaded %d cognate links", len(links))

    sets_path = cog_dir / "cognate_sets.jsonl"
    if sets_path.exists():
        sets_data = []
        with sets_path.open("rb") as fh:
            for line in fh:
                sets_data.append(orjson.loads(line))
        loader.load_cognate_sets(sets_data)
        logger.info("Loaded %d cognate sets", len(sets_data))

    logger.info("Database load complete")
