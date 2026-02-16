"""Placeholder filter hooks for future corpus slicing."""

from __future__ import annotations

from datasets.registry import Corpus


def filter_by_religious_subset(corpus: Corpus) -> Corpus:
    raise NotImplementedError("Religious subset filtering is not implemented yet.")


def filter_by_metadata(corpus: Corpus, predicate) -> Corpus:
    raise NotImplementedError("Metadata filtering is not implemented yet.")
