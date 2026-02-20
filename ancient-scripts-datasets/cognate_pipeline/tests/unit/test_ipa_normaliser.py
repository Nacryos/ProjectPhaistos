"""Tests for phonetic normaliser orchestrator."""

from __future__ import annotations

from cognate_pipeline.config.schema import NormalisationConfig
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.provenance.tracker import ProvenanceRecord


def _make_raw(
    form: str, phonetic_raw: str = "", language_id: str = "eng",
    transcription_type: str = TranscriptionType.UNKNOWN,
) -> RawLexeme:
    return RawLexeme(
        id="test_1",
        language_id=language_id,
        glottocode="",
        concept_id="concept_1",
        form=form,
        phonetic_raw=phonetic_raw,
        transcription_type=transcription_type,
        source_name="test",
        provenance=ProvenanceRecord(
            source_name="test", source_format="test"
        ),
    )


class TestIpaNormaliser:
    def test_attested_ipa_used_first(self):
        """When IPA is already attested, it should be cleaned and used."""
        normaliser = IpaNormaliser()
        raw = _make_raw("water", phonetic_raw="/ˈwɔːtə/", transcription_type=TranscriptionType.IPA)
        result = normaliser.normalise(raw)
        assert result.phonetic_canonical == "ˈwɔːtə"
        assert result.phonetic_raw == "/ˈwɔːtə/"
        assert result.transcription_type == TranscriptionType.IPA
        assert result.confidence == 0.95

    def test_transliteration_passthrough(self):
        """When no IPA and no backend works, form passes through."""
        config = NormalisationConfig(
            ipa_backend_priority=["attested"],
            transliteration_passthrough=True,
        )
        normaliser = IpaNormaliser(config)
        raw = _make_raw("abd", language_id="uga")
        result = normaliser.normalise(raw)
        assert result.phonetic_canonical == "abd"
        assert result.transcription_type == TranscriptionType.TRANSLITERATION
        assert result.confidence == 0.3

    def test_transliteration_type_preserved_from_source(self):
        """When source already declares transliteration, confidence reflects that."""
        normaliser = IpaNormaliser()
        raw = _make_raw(
            "abd", phonetic_raw="abd", language_id="uga",
            transcription_type=TranscriptionType.TRANSLITERATION,
        )
        result = normaliser.normalise(raw)
        assert result.transcription_type == TranscriptionType.TRANSLITERATION
        assert result.confidence == 0.6  # Lower than true IPA

    def test_sound_class_computed(self):
        """Sound class should always be computed from phonetic_canonical."""
        normaliser = IpaNormaliser()
        raw = _make_raw("abd", phonetic_raw="abd", transcription_type=TranscriptionType.IPA)
        result = normaliser.normalise(raw)
        assert result.sound_class != ""
        assert result.sound_class == "ABD"

    def test_provenance_steps_added(self):
        normaliser = IpaNormaliser()
        raw = _make_raw("test", phonetic_raw="tɛst", transcription_type=TranscriptionType.IPA)
        result = normaliser.normalise(raw)
        assert result.provenance is not None
        step_tools = [s.tool for s in result.provenance.steps]
        assert "phonetic_normalise" in step_tools
        assert "sound_class" in step_tools

    def test_empty_phonetic_with_passthrough(self):
        """Form with no phonetic data should use form via passthrough."""
        config = NormalisationConfig(transliteration_passthrough=True)
        normaliser = IpaNormaliser(config)
        raw = _make_raw("msgr", language_id="uga")
        result = normaliser.normalise(raw)
        assert result.phonetic_canonical == "msgr"
        assert result.transcription_type == TranscriptionType.TRANSLITERATION

    def test_strip_suprasegmentals(self):
        config = NormalisationConfig(strip_suprasegmentals=True)
        normaliser = IpaNormaliser(config)
        raw = _make_raw("test", phonetic_raw="/ˈtɛst/", transcription_type=TranscriptionType.IPA)
        result = normaliser.normalise(raw)
        assert "ˈ" not in result.phonetic_canonical

    def test_roundtrip(self):
        """NormalisedLexeme should roundtrip through dict."""
        normaliser = IpaNormaliser()
        raw = _make_raw("ab", phonetic_raw="ab", transcription_type=TranscriptionType.IPA)
        result = normaliser.normalise(raw)
        d = result.to_dict()
        from cognate_pipeline.normalise.models import NormalisedLexeme
        restored = NormalisedLexeme.from_dict(d)
        assert restored.phonetic_canonical == result.phonetic_canonical
        assert restored.sound_class == result.sound_class
        assert restored.transcription_type == result.transcription_type

    def test_provenance_records_transcription_type(self):
        """Provenance should record what transcription type was actually produced."""
        normaliser = IpaNormaliser()
        raw = _make_raw("abd", phonetic_raw="abd", transcription_type=TranscriptionType.TRANSLITERATION)
        result = normaliser.normalise(raw)
        norm_step = [s for s in result.provenance.steps if s.tool == "phonetic_normalise"][0]
        assert norm_step.params["transcription_type"] == TranscriptionType.TRANSLITERATION
