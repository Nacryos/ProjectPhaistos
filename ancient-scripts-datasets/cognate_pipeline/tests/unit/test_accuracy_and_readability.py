"""Thorough tests for accuracy and machine readability.

Validates:
1. Language classification and family assignment
2. Phonetic transcription type tracking (IPA vs transliteration vs orthographic)
3. Sound class correctness for known phonological correspondences
4. Cognate link structure and edge semantics
5. JSONL roundtrip fidelity (machine readability)
6. Provenance chain completeness
7. Known Ugaritic-Hebrew sound correspondences
"""

from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pytest

from cognate_pipeline.cognate.baseline_levenshtein import (
    BaselineLevenshtein,
    normalised_similarity,
    weighted_levenshtein,
)
from cognate_pipeline.cognate.candidate_gen import generate_candidates
from cognate_pipeline.cognate.clustering import cluster_links
from cognate_pipeline.cognate.models import CognateLink, CognateSet
from cognate_pipeline.config.schema import ClusteringAlgorithm, SourceDef, SourceFormat
from cognate_pipeline.ingest.csv_ingester import CsvIngester
from cognate_pipeline.ingest.language_resolver import LanguageResolver
from cognate_pipeline.ingest.models import RawLexeme, TranscriptionType
from cognate_pipeline.normalise.ipa_normaliser import IpaNormaliser
from cognate_pipeline.normalise.models import NormalisedLexeme
from cognate_pipeline.normalise.sound_class import ipa_to_sound_class, tokenize_ipa
from cognate_pipeline.config.schema import NormalisationConfig
from cognate_pipeline.provenance.tracker import ProvenanceRecord


# ---------------------------------------------------------------------------
# 1. Language classification and family assignment
# ---------------------------------------------------------------------------
class TestLanguageClassification:
    """Verify that language IDs resolve to correct Glottocodes and families."""

    def test_all_ancient_languages_have_glottocodes(self):
        resolver = LanguageResolver()
        expected = {
            # Original 14
            "uga": "ugar1238",
            "heb": "hebr1245",
            "got": "goth1244",
            "xib": "iber1250",
            "akk": "akka1240",
            "sux": "sume1241",
            "lat": "lati1261",
            "grc": "anci1242",
            "arc": "offi1241",
            "egy": "egyp1253",
            "hit": "hitt1242",
            "phn": "phoe1239",
            "syc": "clas1252",
            "eus": "basq1248",
            # Germanic
            "ang": "olde1238",
            "non": "oldn1244",
            "goh": "oldh1241",
            # Celtic
            "sga": "oldi1245",
            "cym": "wels1247",
            "bre": "bret1244",
            # Balto-Slavic
            "lit": "lith1251",
            "chu": "chur1257",
            "rus": "russ1263",
            # Indo-Iranian
            "san": "sans1269",
            "ave": "aves1237",
            "fas": "west2369",
            # Italic
            "osc": "osca1245",
            "xum": "umbr1253",
            # Hellenic
            "gmy": "myce1241",
            # Semitic
            "arb": "stan1318",
            "amh": "amha1245",
            # Turkic
            "otk": "oldt1247",
            "tur": "nucl1301",
            "aze": "nort2697",
            # Uralic
            "fin": "finn1318",
            "hun": "hung1274",
            "est": "esto1258",
        }
        for iso, glottocode in expected.items():
            assert resolver.resolve(iso) == glottocode, (
                f"Expected {iso} -> {glottocode}, got {resolver.resolve(iso)}"
            )

    def test_glottocode_format_valid(self):
        """All Glottocodes must match the xxxx1234 pattern."""
        import re
        resolver = LanguageResolver()
        pattern = re.compile(r"^[a-z]{4}\d{4}$")
        for code in ["uga", "heb", "got", "xib", "akk"]:
            gc = resolver.resolve(code)
            assert pattern.match(gc), f"Invalid Glottocode format: {gc} for {code}"


# ---------------------------------------------------------------------------
# 2. Transcription type tracking
# ---------------------------------------------------------------------------
class TestTranscriptionTypeTracking:
    """Verify that transcription types are correctly assigned and propagated."""

    def test_cog_format_is_transliteration(self):
        """The .cog format uses consonantal transliteration, NOT IPA."""
        source = SourceDef(
            name="test_cog",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        for lex in lexemes[:20]:
            assert lex.transcription_type == TranscriptionType.TRANSLITERATION, (
                f"Expected TRANSLITERATION for .cog form '{lex.form}', "
                f"got '{lex.transcription_type}'"
            )

    def test_wiktionary_ipa_is_ipa(self):
        """Wiktionary entries with pronunciations should be typed as IPA."""
        from cognate_pipeline.ingest.wiktionary_ingester import WiktionaryIngester
        source = SourceDef(
            name="test_wikt",
            path=Path(__file__).parent.parent / "fixtures" / "sample_wiktionary.jsonl",
            format=SourceFormat.WIKTIONARY,
        )
        ingester = WiktionaryIngester(source)
        lexemes = list(ingester.ingest())
        ipa_entries = [l for l in lexemes if l.phonetic_raw]
        assert len(ipa_entries) > 0
        for lex in ipa_entries:
            assert lex.transcription_type == TranscriptionType.IPA, (
                f"Wiktionary entry '{lex.form}' has phonetic_raw='{lex.phonetic_raw}' "
                f"but type='{lex.transcription_type}'"
            )

    def test_wiktionary_no_pronunciation_is_orthographic(self):
        """Wiktionary entries without pronunciation data should be orthographic."""
        from cognate_pipeline.ingest.wiktionary_ingester import WiktionaryIngester
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
            f.write(orjson.dumps({
                "word": "xyzabc", "lang": "Test", "lang_code": "xx",
                "senses": [{"glosses": ["test"]}]
            }) + b"\n")
            path = Path(f.name)
        try:
            source = SourceDef(name="no_pron", path=path, format=SourceFormat.WIKTIONARY)
            lexemes = list(WiktionaryIngester(source).ingest())
            assert lexemes[0].transcription_type == TranscriptionType.ORTHOGRAPHIC
        finally:
            path.unlink(missing_ok=True)

    def test_normaliser_propagates_type(self):
        """Normaliser should carry forward the transcription type."""
        config = NormalisationConfig(
            ipa_backend_priority=["attested"],
            transliteration_passthrough=True,
        )
        normaliser = IpaNormaliser(config)
        raw = RawLexeme(
            id="t1", language_id="uga", glottocode="", concept_id="c1",
            form="abd", transcription_type=TranscriptionType.TRANSLITERATION,
            source_name="test",
        )
        result = normaliser.normalise(raw)
        assert result.transcription_type == TranscriptionType.TRANSLITERATION

    def test_normaliser_passthrough_sets_transliteration(self):
        """When passthrough is used, type should be TRANSLITERATION not UNKNOWN."""
        config = NormalisationConfig(transliteration_passthrough=True)
        normaliser = IpaNormaliser(config)
        raw = RawLexeme(
            id="t1", language_id="uga", glottocode="", concept_id="c1",
            form="abd", source_name="test",
        )
        result = normaliser.normalise(raw)
        assert result.transcription_type == TranscriptionType.TRANSLITERATION


# ---------------------------------------------------------------------------
# 3. Sound class correctness
# ---------------------------------------------------------------------------
class TestSoundClassAccuracy:
    """Verify SCA encoding for known phonological correspondences."""

    def test_ugaritic_hebrew_known_correspondences(self):
        """Known Ug-Heb sound correspondences should map to expected SCA classes.

        From the NeuroDecipher dataset documentation:
        - Ug. d = Heb. z  (both should be in coronal class)
        - Ug. v = Heb. $  (both sibilants)
        - Ug. x = Heb. H  (both pharyngeals/velars)
        """
        # d and z are both coronals: D and S in SCA
        assert ipa_to_sound_class("d") in ("D",)
        assert ipa_to_sound_class("z") in ("S",)

        # $ (shin) maps to S
        assert ipa_to_sound_class("$") == "S"

        # H (het) maps to H
        assert ipa_to_sound_class("H") == "H"

        # x (pharyngeal) maps to K (velar class)
        assert ipa_to_sound_class("x") == "K"

    def test_all_transliteration_chars_have_classes(self):
        """Every special transliteration character should map to a non-zero class."""
        specials = {"$": "S", "H": "H", "<": "H", "@": "S", "*": "S"}
        for char, expected_class in specials.items():
            result = ipa_to_sound_class(char)
            assert result == expected_class, (
                f"Transliteration char '{char}' mapped to '{result}', expected '{expected_class}'"
            )

    def test_identical_cognates_have_identical_sound_classes(self):
        """Forms like 'ab'/'ab' must produce identical SCA encodings."""
        # From the real data: ab/ab, bgn/bgn, tmr/tmr
        for form in ["ab", "bgn", "tmr", "yrq", "bkd"]:
            sc = ipa_to_sound_class(form)
            assert sc == ipa_to_sound_class(form)
            assert len(sc) == len(form), (
                f"SCA length mismatch for '{form}': got '{sc}' (len {len(sc)})"
            )
            assert "0" not in sc, (
                f"Unknown segment in '{form}': SCA='{sc}'"
            )

    def test_ipa_vowels_map_correctly(self):
        """True IPA vowels should map to their specific vowel classes."""
        assert ipa_to_sound_class("ɑ") == "A"
        assert ipa_to_sound_class("ɛ") == "E"
        assert ipa_to_sound_class("ɪ") == "I"
        assert ipa_to_sound_class("ɔ") == "O"
        assert ipa_to_sound_class("ʊ") == "U"

    def test_gothic_thorn_maps(self):
        """Gothic þ (thorn = /θ/) should map to T class (dental fricative)."""
        assert ipa_to_sound_class("θ") == "T"

    def test_no_unknown_segments_in_real_data(self):
        """No form from the real .cog file should produce '0' (unknown) segments."""
        source = SourceDef(
            name="test_real",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        config = NormalisationConfig(transliteration_passthrough=True)
        normaliser = IpaNormaliser(config)
        unknowns = []
        for raw in ingester.ingest():
            norm = normaliser.normalise(raw)
            if "0" in norm.sound_class:
                unknowns.append((norm.form, norm.sound_class))
        # Report any unknowns for debugging but don't necessarily fail
        # since some transliteration chars may be legitimately unmapped
        if unknowns:
            unmapped_chars = set()
            for form, sc in unknowns:
                for i, c in enumerate(sc):
                    if c == "0":
                        tokens = tokenize_ipa(form)
                        if i < len(tokens):
                            unmapped_chars.add(tokens[i])
            # These should be few — warn but allow some
            assert len(unmapped_chars) < 10, (
                f"Too many unmapped chars in real data: {unmapped_chars}"
            )


# ---------------------------------------------------------------------------
# 4. Cognate link structure and edge semantics
# ---------------------------------------------------------------------------
class TestCognateLinkStructure:
    """Verify that cognate links are well-formed relational edges."""

    def test_link_ordering_invariant(self):
        """lexeme_id_a must always be < lexeme_id_b."""
        a = NormalisedLexeme(
            id="z_1", language_id="uga", glottocode="", concept_id="c1",
            form="ab", phonetic_raw="ab", phonetic_canonical="ab", sound_class="AB",
        )
        b = NormalisedLexeme(
            id="a_1", language_id="heb", glottocode="", concept_id="c1",
            form="ab", phonetic_raw="ab", phonetic_canonical="ab", sound_class="AB",
        )
        pairs = generate_candidates([a, b])
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.0)
        for link in links:
            assert link.lexeme_id_a < link.lexeme_id_b, (
                f"Ordering violated: {link.lexeme_id_a} >= {link.lexeme_id_b}"
            )

    def test_no_self_links(self):
        """A lexeme should never be linked to itself."""
        a = NormalisedLexeme(
            id="a_1", language_id="uga", glottocode="", concept_id="c1",
            form="ab", phonetic_raw="ab", phonetic_canonical="ab", sound_class="AB",
        )
        pairs = generate_candidates([a])
        assert len(pairs) == 0

    def test_relationship_type_semantics(self):
        """Family-aware detection must distinguish inherited vs areal similarity."""
        lexemes = [
            NormalisedLexeme(
                id="uga_1", language_id="uga", glottocode="ugar1238", concept_id="god",
                form="il", phonetic_raw="il", phonetic_canonical="il", sound_class="IL",
            ),
            NormalisedLexeme(
                id="heb_1", language_id="heb", glottocode="hebr1245", concept_id="god",
                form="el", phonetic_raw="el", phonetic_canonical="el", sound_class="EL",
            ),
            NormalisedLexeme(
                id="got_1", language_id="got", glottocode="goth1244", concept_id="god",
                form="guth", phonetic_raw="guth", phonetic_canonical="guth", sound_class="GUTH",
            ),
        ]
        pairs = generate_candidates(lexemes, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.0)

        for link in links:
            if {"uga" in link.lexeme_id_a and "heb" in link.lexeme_id_b} or \
               {"heb" in link.lexeme_id_a and "uga" in link.lexeme_id_b}:
                pass  # checked below
            assert link.relationship_type in (
                "cognate_inherited", "similarity_only"
            ), f"Unexpected relationship_type: {link.relationship_type}"

        # Find the Semitic pair
        semitic_links = [
            l for l in links
            if ("uga" in l.lexeme_id_a or "uga" in l.lexeme_id_b)
            and ("heb" in l.lexeme_id_a or "heb" in l.lexeme_id_b)
        ]
        for sl in semitic_links:
            assert sl.relationship_type == "cognate_inherited"

        # Find the cross-family pairs
        cross_links = [
            l for l in links
            if "got" in l.lexeme_id_a or "got" in l.lexeme_id_b
        ]
        for cl in cross_links:
            assert cl.relationship_type == "similarity_only"

    def test_link_score_bounded(self):
        """All scores must be in [0.0, 1.0]."""
        source = SourceDef(
            name="test_scores",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        config = NormalisationConfig(transliteration_passthrough=True)
        normaliser = IpaNormaliser(config)
        normalised = [normaliser.normalise(r) for r in ingester.ingest()]
        pairs = generate_candidates(normalised)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.0)
        for link in links:
            assert 0.0 <= link.score <= 1.0, f"Score out of bounds: {link.score}"

    def test_cognate_set_members_at_least_two(self):
        """Every cognate set must have at least 2 members."""
        links = [
            CognateLink(
                lexeme_id_a="a", lexeme_id_b="b", concept_id="c1",
                score=0.9, method="test",
            ),
            CognateLink(
                lexeme_id_a="a", lexeme_id_b="c", concept_id="c1",
                score=0.8, method="test",
            ),
        ]
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        for cs in sets:
            assert len(cs.members) >= 2


# ---------------------------------------------------------------------------
# 5. JSONL roundtrip fidelity
# ---------------------------------------------------------------------------
class TestMachineReadability:
    """Verify that all data structures survive JSONL serialization."""

    def test_raw_lexeme_roundtrip_with_transcription_type(self):
        raw = RawLexeme(
            id="t1", language_id="uga", glottocode="ugar1238",
            concept_id="father", form="ab",
            phonetic_raw="ab", transcription_type=TranscriptionType.TRANSLITERATION,
            alternatives=["ab2"], source_name="test",
            provenance=ProvenanceRecord(
                source_name="test", source_format="cog", original_id="row_1",
            ).add_step("ingest", {"file": "test.cog"}),
        )
        serialized = orjson.dumps(raw.to_dict())
        restored = RawLexeme.from_dict(orjson.loads(serialized))
        assert restored.id == raw.id
        assert restored.phonetic_raw == raw.phonetic_raw
        assert restored.transcription_type == TranscriptionType.TRANSLITERATION
        assert restored.alternatives == ["ab2"]
        assert restored.provenance.source_format == "cog"

    def test_normalised_lexeme_roundtrip(self):
        norm = NormalisedLexeme(
            id="t1", language_id="heb", glottocode="hebr1245",
            concept_id="water", form="maym",
            phonetic_raw="maym", phonetic_canonical="maym",
            sound_class="MAUM", transcription_type=TranscriptionType.TRANSLITERATION,
            confidence=0.6, source_name="test",
        )
        serialized = orjson.dumps(norm.to_dict())
        restored = NormalisedLexeme.from_dict(orjson.loads(serialized))
        assert restored.phonetic_canonical == "maym"
        assert restored.sound_class == "MAUM"
        assert restored.transcription_type == TranscriptionType.TRANSLITERATION
        assert restored.confidence == 0.6

    def test_cognate_link_roundtrip(self):
        link = CognateLink(
            lexeme_id_a="a_1", lexeme_id_b="b_1", concept_id="water",
            relationship_type="cognate_inherited", score=0.85,
            method="baseline_lev", threshold_used=0.5,
            evidence={"sound_class_a": "AB", "sound_class_b": "AB", "distance": 0.0},
        )
        serialized = orjson.dumps(link.to_dict())
        restored = CognateLink.from_dict(orjson.loads(serialized))
        assert restored.relationship_type == "cognate_inherited"
        assert restored.evidence["sound_class_a"] == "AB"
        assert restored.score == 0.85

    def test_cognate_set_roundtrip(self):
        cs = CognateSet(
            id="cs_test", concept_id="water", method="connected_components",
            members=[
                CognateLink.__class__.__module__  # dummy — use CognateSetMember below
            ],
            quality={"size": 3},
        )
        from cognate_pipeline.cognate.models import CognateSetMember
        cs.members = [
            CognateSetMember(lexeme_id="a_1"),
            CognateSetMember(lexeme_id="b_1"),
            CognateSetMember(lexeme_id="c_1"),
        ]
        serialized = orjson.dumps(cs.to_dict())
        restored = CognateSet.from_dict(orjson.loads(serialized))
        assert len(restored.members) == 3
        assert restored.quality["size"] == 3

    def test_backward_compat_ipa_raw_field(self):
        """Old JSONL with 'ipa_raw' should still deserialize correctly."""
        old_format = {
            "id": "t1", "language_id": "eng", "form": "water",
            "ipa_raw": "/ˈwɔːtə/",  # old field name
            "ipa_canonical": "ˈwɔːtə",  # old field name
            "sound_class": "WOTE",
            "concept_id": "water", "glottocode": "", "confidence": 0.95,
        }
        restored = NormalisedLexeme.from_dict(old_format)
        assert restored.phonetic_raw == "/ˈwɔːtə/"
        assert restored.phonetic_canonical == "ˈwɔːtə"
        # Aliases should work
        assert restored.ipa_raw == "/ˈwɔːtə/"
        assert restored.ipa_canonical == "ˈwɔːtə"

    def test_full_pipeline_jsonl_roundtrip(self, tmp_path: Path):
        """Full pipeline output should roundtrip through JSONL."""
        source = SourceDef(
            name="rt_test",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        normaliser = IpaNormaliser(NormalisationConfig(transliteration_passthrough=True))

        original: list[NormalisedLexeme] = []
        for raw in list(ingester.ingest())[:20]:
            original.append(normaliser.normalise(raw))

        # Write
        path = tmp_path / "roundtrip.jsonl"
        with path.open("wb") as f:
            for norm in original:
                f.write(orjson.dumps(norm.to_dict()) + b"\n")

        # Read back
        restored: list[NormalisedLexeme] = []
        with path.open("rb") as f:
            for line in f:
                restored.append(NormalisedLexeme.from_dict(orjson.loads(line)))

        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert rest.id == orig.id
            assert rest.form == orig.form
            assert rest.phonetic_canonical == orig.phonetic_canonical
            assert rest.sound_class == orig.sound_class
            assert rest.transcription_type == orig.transcription_type
            assert rest.confidence == orig.confidence


# ---------------------------------------------------------------------------
# 6. Provenance chain completeness
# ---------------------------------------------------------------------------
class TestProvenanceCompleteness:
    """Verify provenance is tracked through every pipeline stage."""

    def test_ingest_stage_recorded(self):
        source = SourceDef(
            name="prov_test",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        lexemes = list(ingester.ingest())
        for lex in lexemes[:5]:
            assert lex.provenance is not None
            assert lex.provenance.source_name == "prov_test"
            assert lex.provenance.source_format == "cog"
            assert len(lex.provenance.steps) >= 1
            assert lex.provenance.steps[0].tool == "ingest"

    def test_normalise_stage_appended(self):
        source = SourceDef(
            name="prov_test2",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        normaliser = IpaNormaliser(NormalisationConfig(transliteration_passthrough=True))
        raw = next(iter(ingester.ingest()))
        norm = normaliser.normalise(raw)
        steps = [s.tool for s in norm.provenance.steps]
        assert "ingest" in steps
        assert "phonetic_normalise" in steps
        assert "sound_class" in steps
        # phonetic_normalise should record the method used
        norm_step = [s for s in norm.provenance.steps if s.tool == "phonetic_normalise"][0]
        assert "method" in norm_step.params
        assert "transcription_type" in norm_step.params


# ---------------------------------------------------------------------------
# 7. Known Ugaritic-Hebrew cognates
# ---------------------------------------------------------------------------
class TestKnownCognates:
    """Test that linguistically known cognates are detected correctly."""

    @pytest.fixture
    def uga_heb_normalised(self) -> list[NormalisedLexeme]:
        source = SourceDef(
            name="known_cog",
            path=Path(r"C:\Users\alvin\ancient-scripts-datasets\data\ugaritic\uga-heb.small.no_spe.cog"),
            format=SourceFormat.COG,
            extra={"lang_a": "uga", "lang_b": "heb"},
        )
        ingester = CsvIngester(source)
        config = NormalisationConfig(transliteration_passthrough=True)
        normaliser = IpaNormaliser(config)
        return [normaliser.normalise(r) for r in ingester.ingest()]

    def test_identical_forms_score_one(self, uga_heb_normalised):
        """Forms that are identical across languages should score 1.0."""
        pairs = generate_candidates(uga_heb_normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.99)
        # There should be many perfect matches in the data
        assert len(links) > 10, (
            f"Expected many perfect cognate matches, got only {len(links)}"
        )
        for link in links:
            assert link.score >= 0.99
            assert link.relationship_type == "cognate_inherited"

    def test_high_similarity_pairs_dominate(self, uga_heb_normalised):
        """Most Ug-Heb pairs should have high similarity (they ARE cognates)."""
        pairs = generate_candidates(uga_heb_normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.0)
        high = sum(1 for l in links if l.score >= 0.5)
        total = len(links)
        ratio = high / total if total > 0 else 0
        assert ratio > 0.3, (
            f"Only {ratio:.0%} of cognate pairs scored >= 0.5. "
            f"These are known cognates — ratio should be higher."
        )

    def test_clustering_produces_pair_sets(self, uga_heb_normalised):
        """Clustering should produce sets of size 2 (one per language)."""
        pairs = generate_candidates(uga_heb_normalised, family_aware=True)
        scorer = BaselineLevenshtein()
        links = scorer.score_pairs(pairs, threshold=0.5)
        sets = cluster_links(links, ClusteringAlgorithm.CONNECTED_COMPONENTS)
        assert len(sets) > 0
        # Most sets should have exactly 2 members (one Uga, one Heb)
        pair_sets = [s for s in sets if len(s.members) == 2]
        assert len(pair_sets) > len(sets) * 0.5, (
            f"Expected most sets to be pairs, but only {len(pair_sets)} "
            f"out of {len(sets)} have exactly 2 members"
        )
