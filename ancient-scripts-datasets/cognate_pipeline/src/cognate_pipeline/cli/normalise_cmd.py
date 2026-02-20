"""CLI handler for the normalise-ipa subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

import orjson

from cognate_pipeline.config.loader import load_config
from cognate_pipeline.ingest.models import RawLexeme
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_normalise(config_path: str) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)

    ingest_dir = cfg.staging_dir / "ingest"
    norm_dir = cfg.staging_dir / "normalised"
    norm_dir.mkdir(parents=True, exist_ok=True)

    normaliser = IpaNormaliser(cfg.normalisation)

    for jsonl_path in sorted(ingest_dir.glob("*.jsonl")):
        if jsonl_path.name.startswith("_"):
            continue
        logger.info("Normalising %s", jsonl_path.name)
        out_path = norm_dir / jsonl_path.name
        count = 0
        with jsonl_path.open("rb") as fin, out_path.open("wb") as fout:
            for line in fin:
                raw = RawLexeme.from_dict(orjson.loads(line))
                normalised = normaliser.normalise(raw)
                fout.write(orjson.dumps(normalised.to_dict()) + b"\n")
                count += 1
        logger.info("  Wrote %d normalised lexemes to %s", count, out_path)
