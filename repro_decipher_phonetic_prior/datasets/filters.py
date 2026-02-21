"""Filter hooks for corpus slicing.

Religious subsets are now handled via the registry's variant system:
    get_corpus("gothic", variant="religious")
    get_corpus("ugaritic", variant="religious")
    get_corpus("iberian", variant="religious")

These functions provide programmatic filtering for ad-hoc subsets.
"""

from __future__ import annotations

from typing import Callable

from datasets.registry import Corpus, get_corpus


def filter_by_religious_subset(corpus_name: str) -> Corpus:
    """Load the religious-text variant of a corpus via the registry."""
    return get_corpus(corpus_name, variant="religious")


def filter_by_metadata(corpus: Corpus, predicate: Callable) -> Corpus:
    """Filter a corpus's lost_text by a metadata predicate on ground_truth rows."""
    if not corpus.ground_truth:
        return corpus
    kept_indices = set()
    kept_lost: list[str] = []
    for i, gt_row in enumerate(corpus.ground_truth):
        if i < len(corpus.lost_text) and predicate(gt_row):
            kept_indices.add(i)
            kept_lost.append(corpus.lost_text[i])
    from datasets.registry import _deterministic_splits
    return Corpus(
        name=corpus.name,
        variant=corpus.variant,
        lost_text=kept_lost,
        known_text=corpus.known_text,
        metadata={**corpus.metadata, "filtered": True},
        splits=_deterministic_splits(kept_lost),
        ground_truth=[gt for i, gt in enumerate(corpus.ground_truth) if i in kept_indices],
    )
