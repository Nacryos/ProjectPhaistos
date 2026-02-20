"""Base protocol for source ingesters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from cognate_pipeline.config.schema import SourceDef
from cognate_pipeline.ingest.models import RawLexeme


@runtime_checkable
class SourceIngester(Protocol):
    """Protocol that all ingesters must implement."""

    def __init__(self, source_def: SourceDef) -> None: ...

    def ingest(self) -> Iterator[RawLexeme]:
        """Yield RawLexeme objects from the source."""
        ...
