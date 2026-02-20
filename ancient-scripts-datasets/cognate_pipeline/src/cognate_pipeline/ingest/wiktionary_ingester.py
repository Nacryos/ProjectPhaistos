"""Wiktextract JSONL ingester."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import orjson

from cognate_pipeline.config.schema import SourceDef
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.provenance.tracker import ProvenanceRecord

logger = logging.getLogger(__name__)


class WiktionaryIngester:
    """Ingests Wiktextract JSONL exports.

    Each line is a JSON object with at minimum:
      - word: the headword
      - lang: language name
      - lang_code: Wiktionary language code
      - pronunciations: list of {ipa: "...", ...}
      - senses: list of {glosses: [...], ...}
    """

    def __init__(self, source_def: SourceDef) -> None:
        self.source_def = source_def

    def ingest(self) -> Iterator[RawLexeme]:
        path = Path(self.source_def.path)
        with path.open("rb") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = orjson.loads(line)
                except orjson.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num)
                    continue

                word = entry.get("word", "").strip()
                if not word:
                    continue

                lang = entry.get("lang", "")
                lang_code = entry.get("lang_code", "")

                # Extract IPA from pronunciations (Wiktionary provides true IPA)
                phonetic_raw = ""
                transcription_type = TranscriptionType.ORTHOGRAPHIC
                pronunciations = entry.get("pronunciations", entry.get("sounds", []))
                if isinstance(pronunciations, list):
                    for pron in pronunciations:
                        if isinstance(pron, dict) and pron.get("ipa"):
                            phonetic_raw = pron["ipa"]
                            transcription_type = TranscriptionType.IPA
                            break

                # Extract concept from first sense gloss
                concept_id = ""
                senses = entry.get("senses", [])
                if senses and isinstance(senses, list):
                    first_sense = senses[0]
                    if isinstance(first_sense, dict):
                        glosses = first_sense.get("glosses", [])
                        if glosses:
                            concept_id = glosses[0]

                # Extract etymology info
                etymology = entry.get("etymology_text", "")

                yield RawLexeme(
                    id=f"{self.source_def.name}_{line_num}",
                    language_id=lang_code or lang,
                    glottocode="",
                    concept_id=concept_id,
                    form=word,
                    phonetic_raw=phonetic_raw,
                    transcription_type=transcription_type,
                    source_name=self.source_def.name,
                    provenance=ProvenanceRecord(
                        source_name=self.source_def.name,
                        source_format="wiktionary",
                        original_id=f"line_{line_num}",
                    ).add_step(
                        "ingest",
                        {"file": path.name, "line": line_num},
                    ),
                    extra={"etymology": etymology} if etymology else {},
                )
