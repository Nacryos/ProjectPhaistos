"""CLDF FormTable ingester using pycldf."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from cognate_pipeline.config.schema import SourceDef
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.provenance.tracker import ProvenanceRecord

logger = logging.getLogger(__name__)


class CldfIngester:
    """Ingests CLDF Wordlist datasets via pycldf."""

    def __init__(self, source_def: SourceDef) -> None:
        self.source_def = source_def

    def ingest(self) -> Iterator[RawLexeme]:
        try:
            from pycldf import Dataset
        except ImportError:
            raise ImportError(
                "pycldf is required for CLDF ingestion. "
                "Install with: pip install cognate-pipeline[cldf]"
            )

        path = self.source_def.path
        # Find metadata file
        metadata_path = path / "Wordlist-metadata.json"
        if not metadata_path.exists():
            # Try cldf-metadata.json
            metadata_path = path / "cldf-metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No CLDF metadata found in {path}. "
                "Expected Wordlist-metadata.json or cldf-metadata.json"
            )

        ds = Dataset.from_metadata(metadata_path)

        # Build language_id -> glottocode mapping
        lang_map: dict[str, str] = {}
        if "LanguageTable" in ds:
            for lang in ds["LanguageTable"]:
                lid = lang.get("ID", "")
                glottocode = lang.get("Glottocode", "")
                lang_map[lid] = glottocode

        # Read forms
        for form in ds["FormTable"]:
            form_id = str(form.get("ID", ""))
            language_id = str(form.get("Language_ID", ""))
            concept_id = str(form.get("Parameter_ID", ""))
            value = str(form.get("Form", ""))

            if not value or value == "_":
                continue

            glottocode = lang_map.get(language_id, "")

            # Segments in CLDF are typically IPA tokenisations
            segments = form.get("Segments", [])
            phonetic_raw = ""
            transcription_type = TranscriptionType.UNKNOWN
            if segments:
                if isinstance(segments, list):
                    phonetic_raw = " ".join(segments)
                else:
                    phonetic_raw = str(segments)
                transcription_type = TranscriptionType.IPA

            yield RawLexeme(
                id=form_id or f"{self.source_def.name}_{language_id}_{concept_id}",
                language_id=language_id,
                glottocode=glottocode,
                concept_id=concept_id,
                form=value,
                phonetic_raw=phonetic_raw,
                transcription_type=transcription_type,
                source_name=self.source_def.name,
                provenance=ProvenanceRecord(
                    source_name=self.source_def.name,
                    source_format="cldf",
                    original_id=form_id,
                ).add_step("ingest", {"dataset": str(path)}),
            )
