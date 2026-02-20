"""Generic JSON/NDJSON ingester with field mapping."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import orjson

from cognate_pipeline.config.schema import SourceDef, SourceFormat
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.provenance.tracker import ProvenanceRecord

logger = logging.getLogger(__name__)


class JsonIngester:
    """Ingests JSON arrays or NDJSON files with configurable field mapping."""

    def __init__(self, source_def: SourceDef) -> None:
        self.source_def = source_def

    def ingest(self) -> Iterator[RawLexeme]:
        path = Path(self.source_def.path)
        if self.source_def.format == SourceFormat.NDJSON:
            yield from self._ingest_ndjson(path)
        else:
            yield from self._ingest_json_array(path)

    def _ingest_ndjson(self, path: Path) -> Iterator[RawLexeme]:
        with path.open("rb") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num)
                    continue
                lexeme = self._map_object(obj, line_num)
                if lexeme:
                    yield lexeme

    def _ingest_json_array(self, path: Path) -> Iterator[RawLexeme]:
        data = orjson.loads(path.read_bytes())
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try to find a list field
            for key in ("forms", "data", "entries", "items", "results"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else:
                items = [data]
        else:
            return

        for idx, obj in enumerate(items, start=1):
            lexeme = self._map_object(obj, idx)
            if lexeme:
                yield lexeme

    def _map_object(self, obj: dict, idx: int) -> RawLexeme | None:
        mapping = self.source_def.column_mapping
        form = self._get_nested(obj, mapping.form) or ""
        if not form or form == "_":
            return None

        language_id = self._get_nested(obj, mapping.language) or ""
        concept_id = self._get_nested(obj, mapping.concept) or ""
        glottocode = ""
        if mapping.glottocode:
            glottocode = self._get_nested(obj, mapping.glottocode) or ""
        phonetic_raw = ""
        transcription_type = TranscriptionType.UNKNOWN
        if mapping.ipa:
            phonetic_raw = self._get_nested(obj, mapping.ipa) or ""
            if phonetic_raw:
                transcription_type = TranscriptionType.IPA

        return RawLexeme(
            id=f"{self.source_def.name}_{idx}",
            language_id=language_id,
            glottocode=glottocode,
            concept_id=concept_id,
            form=form,
            phonetic_raw=phonetic_raw,
            transcription_type=transcription_type,
            source_name=self.source_def.name,
            provenance=ProvenanceRecord(
                source_name=self.source_def.name,
                source_format=self.source_def.format.value,
                original_id=str(idx),
            ).add_step("ingest", {"index": idx}),
        )

    @staticmethod
    def _get_nested(obj: dict, path: str) -> str:
        """Resolve a dot-separated path in a nested dict."""
        if not path:
            return ""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return ""
            if current is None:
                return ""
        return str(current)
