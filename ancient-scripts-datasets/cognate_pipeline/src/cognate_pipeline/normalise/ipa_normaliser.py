"""Phonetic normalisation orchestrator.

Tries each backend in priority order:
1. attested — use the phonetic representation already present (from source)
2. epitran — grapheme-to-phoneme via Epitran (produces true IPA)
3. phonemizer — espeak-ng fallback (produces true IPA)
4. transliteration_passthrough — use the raw form as-is (NOT IPA)

Records every step in provenance, including what type of transcription
was actually produced.
"""

from __future__ import annotations

import logging

from cognate_pipeline.config.schema import NormalisationConfig
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.normalise import unicode_cleanup
from cognate_pipeline.normalise.epitran_backend import transliterate as epitran_transliterate
from cognate_pipeline.normalise.phonemizer_backend import phonemize
from cognate_pipeline.normalise.sound_class import ipa_to_sound_class
from cognate_pipeline.normalise.models import NormalisedLexeme
from cognate_pipeline.provenance.tracker import ProvenanceRecord

logger = logging.getLogger(__name__)


class IpaNormaliser:
    """Orchestrates phonetic normalisation across multiple backends."""

    def __init__(self, config: NormalisationConfig | None = None) -> None:
        self.config = config or NormalisationConfig()

    def normalise(self, raw: RawLexeme) -> NormalisedLexeme:
        """Normalise a single RawLexeme into a NormalisedLexeme."""
        provenance = raw.provenance or ProvenanceRecord(
            source_name=raw.source_name, source_format="unknown"
        )

        phonetic_raw = raw.phonetic_raw
        phonetic_canonical = ""
        confidence = 1.0
        method_used = "none"
        # Carry forward the source transcription type; backends may upgrade it
        transcription_type = raw.transcription_type

        for backend in self.config.ipa_backend_priority:
            if backend == "attested" and phonetic_raw:
                phonetic_canonical = unicode_cleanup.full_cleanup(
                    phonetic_raw,
                    unicode_form=self.config.unicode_form,
                    strip_supra=self.config.strip_suprasegmentals,
                    strip_ws=self.config.strip_whitespace,
                )
                method_used = "attested"
                # If source said it was IPA, trust that; otherwise keep its type
                if transcription_type == TranscriptionType.IPA:
                    confidence = 0.95
                elif transcription_type == TranscriptionType.TRANSLITERATION:
                    confidence = 0.6
                else:
                    confidence = 0.8
                break

            elif backend == "epitran" and not phonetic_canonical:
                result = epitran_transliterate(raw.form, raw.language_id)
                if result:
                    phonetic_canonical = unicode_cleanup.full_cleanup(
                        result,
                        unicode_form=self.config.unicode_form,
                        strip_supra=self.config.strip_suprasegmentals,
                        strip_ws=self.config.strip_whitespace,
                    )
                    method_used = "epitran"
                    transcription_type = TranscriptionType.IPA  # Epitran produces IPA
                    confidence = 0.7
                    break

            elif backend == "phonemizer" and not phonetic_canonical:
                result = phonemize(raw.form, raw.language_id)
                if result:
                    phonetic_canonical = unicode_cleanup.full_cleanup(
                        result,
                        unicode_form=self.config.unicode_form,
                        strip_supra=self.config.strip_suprasegmentals,
                        strip_ws=self.config.strip_whitespace,
                    )
                    method_used = "phonemizer"
                    transcription_type = TranscriptionType.IPA  # phonemizer produces IPA
                    confidence = 0.5
                    break

        # For transliteration-based scripts, pass through the form as-is
        if not phonetic_canonical and self.config.transliteration_passthrough:
            phonetic_canonical = unicode_cleanup.full_cleanup(
                raw.form,
                unicode_form=self.config.unicode_form,
                strip_ws=self.config.strip_whitespace,
            )
            method_used = "transliteration_passthrough"
            transcription_type = TranscriptionType.TRANSLITERATION
            confidence = 0.3

        provenance.add_step(
            "phonetic_normalise",
            {
                "method": method_used,
                "transcription_type": transcription_type,
                "backend_priority": self.config.ipa_backend_priority,
            },
            result=phonetic_canonical[:50] if phonetic_canonical else "empty",
        )

        # Compute sound class
        sound_class = ipa_to_sound_class(phonetic_canonical)
        provenance.add_step("sound_class", {}, result=sound_class)

        return NormalisedLexeme(
            id=raw.id,
            language_id=raw.language_id,
            glottocode=raw.glottocode,
            concept_id=raw.concept_id,
            form=raw.form,
            phonetic_raw=phonetic_raw,
            phonetic_canonical=phonetic_canonical,
            sound_class=sound_class,
            transcription_type=transcription_type,
            confidence=confidence,
            alternatives=raw.alternatives,
            source_name=raw.source_name,
            provenance=provenance,
            extra=raw.extra,
        )
