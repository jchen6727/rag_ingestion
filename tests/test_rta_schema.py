"""
Invariants for the ACTIVE RTA schema (config/rta_v1.json) and its alignment with
the ChunkMetadata model and the schema-driven loader.

These lock in the 2026-07-20 migration from the 37-field unified schema
(metadata_schema.json, retained for ASA) to the 23-field RTA schema. They catch
the silent-failure classes called out in summary.md / schema_recommendations.md:
schema drift, an unconstrained applies_when vocabulary, a missing
directionality/applies_when pairing, and stale prompt-facing notes.

Run:
    pytest tests/test_rta_schema.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.schema_loader import SchemaVocabulary
from models import ChunkMetadata

_RTA_PATH = Path(__file__).parent.parent / "config" / "rta_v1.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    with open(_RTA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def vocab() -> SchemaVocabulary:
    return SchemaVocabulary.from_path(_RTA_PATH)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestShape:
    def test_field_count_is_23(self, schema: dict) -> None:
        assert len(schema["properties"]) == 23

    def test_title_is_v1(self, schema: dict) -> None:
        assert schema["title"] == "ChunkMetadata_v1"

    def test_rta_only_fields_present(self, schema: dict) -> None:
        props = schema["properties"]
        for f in ("directionality", "applies_when", "clinical_measure_tags"):
            assert f in props

    def test_asa_only_fields_absent(self, schema: dict) -> None:
        """Fields split out to metadata_schema.json must not reappear here."""
        props = schema["properties"]
        for f in (
            "corpus_scope", "analysis_function", "evidence_base", "time_horizon",
            "target_audience", "practice_recommendation_level", "missingness",
            "outcome_measure_tags", "sample_size",
        ):
            assert f not in props


# ---------------------------------------------------------------------------
# Defaults & required
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_directionality_has_default(self, schema: dict, vocab: SchemaVocabulary) -> None:
        assert schema["properties"]["directionality"]["default"] == "neutral"
        assert vocab.default("directionality") == "neutral"

    def test_domain_default_is_general_bucket(self, vocab: SchemaVocabulary) -> None:
        # rta_v1 domain enum has no "other"; the umbrella bucket is the fallback.
        assert vocab.default("domain") == "psychotherapy_general"

    def test_required_provenance_fields(self, schema: dict) -> None:
        assert set(schema["required"]) == {
            "doc_id", "source_file", "domain", "doc_type",
            "page_start", "page_end", "chunk_index",
        }


# ---------------------------------------------------------------------------
# applies_when controlled vocabulary (highest-severity gap in summary.md §2.3)
# ---------------------------------------------------------------------------


class TestAppliesWhenVocabulary:
    def test_applies_when_is_constrained(self, vocab: SchemaVocabulary) -> None:
        assert vocab.enum_values("applies_when") is not None  # not free-text

    def test_applies_when_equals_union(self, schema: dict) -> None:
        props = schema["properties"]
        events = {v for v in props["session_event_tags"]["items"]["enum"] if v != "none"}
        pres = {v for v in props["clinical_presentation"]["items"]["enum"] if v != "other"}
        states = set(schema["$defs"]["state_vocab"]["enum"])
        applies_when = set(props["applies_when"]["items"]["enum"])
        assert applies_when == events | pres | states

    def test_state_vocab_defined(self, schema: dict) -> None:
        states = schema["$defs"]["state_vocab"]["enum"]
        assert "acute_suicidality" in states
        assert "intoxication" in states


# ---------------------------------------------------------------------------
# directionality / applies_when pairing conditional
# ---------------------------------------------------------------------------


class TestPairingConditional:
    def test_if_then_present(self, schema: dict) -> None:
        assert schema["if"]["properties"]["directionality"]["enum"] == [
            "contraindicated", "cautionary",
        ]
        assert schema["then"]["properties"]["applies_when"]["minItems"] == 1


# ---------------------------------------------------------------------------
# Stale-reference hygiene (descriptions are read into the ingestion prompt)
# ---------------------------------------------------------------------------


class TestNoStaleReferences:
    def test_obsolete_notes_deleted(self, schema: dict) -> None:
        notes = schema.get("notes", {})
        for gone in (
            "routing_safety",
            "representation_and_implementation_fields",
            "recommended_change_items",
        ):
            assert gone not in notes

    def test_columbia_normalized_to_cssrs(self, schema: dict) -> None:
        measures = schema["properties"]["clinical_measure_tags"]["items"]["enum"]
        assert "Columbia" not in measures
        assert "C-SSRS" in measures

    def test_descriptions_do_not_reference_removed_fields(self, schema: dict) -> None:
        for name, prop in schema["properties"].items():
            desc = prop.get("description", "")
            assert "evidence_base" not in desc, name
            assert "analysis_function" not in desc, name
            assert "pre_intake_consultation" not in desc, name


# ---------------------------------------------------------------------------
# Model <-> schema alignment (uploader serializes ChunkMetadata to structData)
# ---------------------------------------------------------------------------


class TestModelAlignment:
    def test_model_fields_match_schema(self, schema: dict) -> None:
        assert set(ChunkMetadata.model_fields) == set(schema["properties"])

    def test_model_roundtrips_coerced_dict(self) -> None:
        vocab = SchemaVocabulary.from_path(_RTA_PATH)
        raw = {
            "domain": "cognitive_behavioral_therapy",
            "doc_type": "treatment_manual",
            "therapeutic_modality": ["PE", "CPT"],
            "directionality": "contraindicated",
            "applies_when": ["acute_suicidality", "not_a_state"],  # invalid dropped
            "clinical_measure_tags": ["Columbia", "PCL-5"],        # Columbia dropped
        }
        md = ChunkMetadata(doc_id="a" * 64, **vocab.coerce(raw))
        assert md.applies_when == ["acute_suicidality"]
        assert md.clinical_measure_tags == ["PCL-5"]
        assert set(md.model_dump()) == set(vocab.field_names)
