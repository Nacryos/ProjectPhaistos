"""Utilities for batched iteration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Yield successive batches of size *n* from *iterable*."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch
