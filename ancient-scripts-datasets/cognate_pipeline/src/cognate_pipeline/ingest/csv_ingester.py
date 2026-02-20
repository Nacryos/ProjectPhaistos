"""CSV/TSV/COG file ingester with configurable column mapping."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path

from cognate_pipeline.config.schema import SourceDef, SourceFormat
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.provenance.tracker import ProvenanceRecord

logger = logging.getLogger(__name__)


class CsvIngester:
    """Ingests CSV, TSV, and .cog (cognate pair) files."""

    def __init__(self, source_def: SourceDef) -> None:
        self.source_def = source_def

    def ingest(self) -> Iterator[RawLexeme]:
        fmt = self.source_def.format
        if fmt in (SourceFormat.COG, SourceFormat.TSV):
            delimiter = self.source_def.delimiter or "\t"
        else:
            delimiter = self.source_def.delimiter or ","

        path = Path(self.source_def.path)
        if fmt == SourceFormat.COG:
            yield from self._ingest_cog(path, delimiter)
        else:
            yield from self._ingest_tabular(path, delimiter)

    def _ingest_cog(self, path: Path, delimiter: str) -> Iterator[RawLexeme]:
        """Ingest .cog format: two-column cognate pair file.

        Each row has (lang_a_form, lang_b_form).
        lang_b_form may contain pipe-separated alternatives.
        `_` represents null / missing.

        These files use consonantal transliteration, NOT IPA.
        """
        extra = self.source_def.extra
        lang_a = extra.get("lang_a", "lang_a")
        lang_b = extra.get("lang_b", "lang_b")

        with path.open("r", encoding=self.source_def.encoding) as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return

            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 2:
                    continue
                form_a = row[0].strip()
                form_b_raw = row[1].strip()

                # Skip null entries
                if form_a == "_" or not form_a:
                    continue

                # Generate a concept_id from the pair (the cognate pair itself)
                concept_id = f"pair_{row_idx}"

                # Emit lang_a form
                yield RawLexeme(
                    id=f"{self.source_def.name}_{lang_a}_{row_idx}",
                    language_id=lang_a,
                    glottocode="",
                    concept_id=concept_id,
                    form=form_a,
                    transcription_type=TranscriptionType.TRANSLITERATION,
                    source_name=self.source_def.name,
                    provenance=ProvenanceRecord(
                        source_name=self.source_def.name,
                        source_format="cog",
                        original_id=f"row_{row_idx}_col_a",
                    ).add_step("ingest", {"file": path.name, "row": row_idx}),
                )

                # Emit lang_b forms (pipe-separated alternatives)
                if form_b_raw == "_" or not form_b_raw:
                    continue
                alternatives = [f.strip() for f in form_b_raw.split("|") if f.strip() and f.strip() != "_"]
                if not alternatives:
                    continue

                primary = alternatives[0]
                rest = alternatives[1:] if len(alternatives) > 1 else []
                yield RawLexeme(
                    id=f"{self.source_def.name}_{lang_b}_{row_idx}",
                    language_id=lang_b,
                    glottocode="",
                    concept_id=concept_id,
                    form=primary,
                    alternatives=rest,
                    transcription_type=TranscriptionType.TRANSLITERATION,
                    source_name=self.source_def.name,
                    provenance=ProvenanceRecord(
                        source_name=self.source_def.name,
                        source_format="cog",
                        original_id=f"row_{row_idx}_col_b",
                    ).add_step("ingest", {"file": path.name, "row": row_idx}),
                )

    def _ingest_tabular(self, path: Path, delimiter: str) -> Iterator[RawLexeme]:
        """Ingest standard CSV/TSV with column mapping."""
        mapping = self.source_def.column_mapping

        with path.open("r", encoding=self.source_def.encoding) as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row_idx, row in enumerate(reader, start=1):
                form_col = mapping.form or "Form"
                form = row.get(form_col, "").strip()
                if not form or form == "_":
                    continue

                lang_col = mapping.language or "Language_ID"
                language_id = row.get(lang_col, "").strip()

                concept_col = mapping.concept or "Parameter_ID"
                concept_id = row.get(concept_col, "").strip()

                glottocode = ""
                if mapping.glottocode:
                    glottocode = row.get(mapping.glottocode, "").strip()

                phonetic_raw = ""
                transcription_type = TranscriptionType.UNKNOWN
                if mapping.ipa:
                    phonetic_raw = row.get(mapping.ipa, "").strip()
                    if phonetic_raw:
                        transcription_type = TranscriptionType.IPA

                # Handle pipe-separated alternatives in form
                alternatives = []
                if "|" in form:
                    parts = [p.strip() for p in form.split("|") if p.strip()]
                    form = parts[0]
                    alternatives = parts[1:]

                source_id = ""
                if mapping.source_id:
                    source_id = row.get(mapping.source_id, "").strip()

                yield RawLexeme(
                    id=source_id or f"{self.source_def.name}_{row_idx}",
                    language_id=language_id,
                    glottocode=glottocode,
                    concept_id=concept_id,
                    form=form,
                    phonetic_raw=phonetic_raw,
                    transcription_type=transcription_type,
                    alternatives=alternatives,
                    source_name=self.source_def.name,
                    provenance=ProvenanceRecord(
                        source_name=self.source_def.name,
                        source_format=self.source_def.format.value,
                        original_id=source_id or f"row_{row_idx}",
                    ).add_step("ingest", {"file": path.name, "row": row_idx}),
                )
