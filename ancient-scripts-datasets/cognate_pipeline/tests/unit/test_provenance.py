"""Tests for provenance tracking."""

from __future__ import annotations

from cognate_pipeline.provenance.tracker import ProvenanceRecord, ProvenanceStep
from cognate_pipeline.provenance.license_registry import LicenseRegistry


class TestProvenanceStep:
    def test_roundtrip(self):
        step = ProvenanceStep(tool="epitran", params={"lang": "uga"}, result="ok")
        d = step.to_dict()
        restored = ProvenanceStep.from_dict(d)
        assert restored.tool == "epitran"
        assert restored.params == {"lang": "uga"}
        assert restored.result == "ok"
        assert restored.timestamp == step.timestamp


class TestProvenanceRecord:
    def test_chain_steps(self):
        rec = ProvenanceRecord(source_name="test", source_format="csv")
        rec.add_step("ingest", {"file": "a.csv"}, "15 rows")
        rec.add_step("nfc_normalize", {}, "done")
        assert len(rec.steps) == 2
        assert rec.steps[0].tool == "ingest"
        assert rec.steps[1].tool == "nfc_normalize"

    def test_fluent_api(self):
        rec = (
            ProvenanceRecord(source_name="src", source_format="cldf")
            .add_step("ingest")
            .add_step("normalize")
        )
        assert len(rec.steps) == 2

    def test_roundtrip(self):
        rec = ProvenanceRecord(
            source_name="src", source_format="tsv", original_id="row42"
        )
        rec.add_step("ingest", {"col": "form"})
        d = rec.to_dict()
        restored = ProvenanceRecord.from_dict(d)
        assert restored.source_name == "src"
        assert restored.original_id == "row42"
        assert len(restored.steps) == 1


class TestLicenseRegistry:
    def test_register_and_get(self):
        reg = LicenseRegistry()
        reg.register("src1", "CC-BY-4.0", "https://example.com/license")
        entry = reg.get("src1")
        assert entry is not None
        assert entry.license == "CC-BY-4.0"

    def test_missing_returns_none(self):
        reg = LicenseRegistry()
        assert reg.get("nonexistent") is None

    def test_roundtrip(self):
        reg = LicenseRegistry()
        reg.register("a", "MIT", citation_bibtex="@misc{a}")
        reg.register("b", "CC0")
        d = reg.to_dict()
        restored = LicenseRegistry.from_dict(d)
        assert restored.get("a").license == "MIT"
        assert restored.get("b").license == "CC0"
        assert restored.get("a").citation_bibtex == "@misc{a}"
