"""CLI handler for the detect-cognates subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

import orjson

from cognate_pipeline.config.loader import load_config
from cognate_pipeline.config.schema import CognateMethod
from cognate_pipeline.cognate.candidate_gen import generate_candidates
from cognate_pipeline.cognate.baseline_levenshtein import BaselineLevenshtein
from cognate_pipeline.cognate.lexstat_detector import LexStatDetector
from cognate_pipeline.cognate.clustering import cluster_links
from cognate_pipeline.normalise.models import NormalisedLexeme
from cognate_pipeline.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_detect(config_path: str, method_override: str | None) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.log_level)

    method = CognateMethod(method_override) if method_override else cfg.cognate.method

    norm_dir = cfg.staging_dir / "normalised"
    cog_dir = cfg.staging_dir / "cognate"
    cog_dir.mkdir(parents=True, exist_ok=True)

    # Load all normalised lexemes
    lexemes: list[NormalisedLexeme] = []
    for jsonl_path in sorted(norm_dir.glob("*.jsonl")):
        if jsonl_path.name.startswith("_"):
            continue
        with jsonl_path.open("rb") as fh:
            for line in fh:
                lexemes.append(NormalisedLexeme.from_dict(orjson.loads(line)))
    logger.info("Loaded %d normalised lexemes", len(lexemes))

    # Generate candidate pairs
    pairs = generate_candidates(lexemes, family_aware=cfg.cognate.family_aware)
    logger.info("Generated %d candidate pairs", len(pairs))

    # Score pairs
    if method == CognateMethod.BASELINE_LEV:
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=cfg.cognate.threshold)
    elif method == CognateMethod.LEXSTAT:
        detector = LexStatDetector()
        links = detector.detect(lexemes, threshold=cfg.cognate.threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    logger.info("Scored %d cognate links above threshold", len(links))

    # Cluster
    sets = cluster_links(links, algorithm=cfg.cognate.clustering)
    logger.info("Formed %d cognate sets", len(sets))

    # Write links
    links_path = cog_dir / "cognate_links.jsonl"
    with links_path.open("wb") as fh:
        for link in links:
            fh.write(orjson.dumps(link.to_dict()) + b"\n")

    # Write sets
    sets_path = cog_dir / "cognate_sets.jsonl"
    with sets_path.open("wb") as fh:
        for cset in sets:
            fh.write(orjson.dumps(cset.to_dict()) + b"\n")

    logger.info("Cognate results written to %s", cog_dir)
