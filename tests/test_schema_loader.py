"""
Tests for config/schema_loader.py.

Verifies that SchemaVocabulary derives the controlled vocabulary, defaults, and
coercion behavior from config/metadata_schema.json (the authoritative
psychotherapy schema) — the single source of truth that ingestion/metadata_gen.py
and scripts/setup_vertex_search.py both consume.

Run:
    pytest tests/test_schema_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.schema_loader import SchemaVocabulary, load_vocabulary

_SCHEMA_PATH = Path(__file__).parent.parent / "config" / "metadata_schema.json"


@pytest.fixture(scope="module")
def vocab() -> SchemaVocabulary:
    return SchemaVocabulary.from_path(_SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Loading / introspection
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_loads_from_default_path(self) -> None:
        """from_path() with no arg resolves to config/metadata_schema.json."""
        v = SchemaVocabulary.from_path()
        assert "domain" in v.field_names

    def test_load_vocabulary_helper(self) -> None:
        assert isinstance(load_vocabulary(_SCHEMA_PATH), SchemaVocabulary)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SchemaVocabulary.from_path(tmp_path / "does_not_exist.json")

    def test_domain_enum_is_psychotherapy(self, vocab: SchemaVocabulary) -> None:
        """The domain enum is the psychotherapy vocabulary, not biomedical."""
        domains = vocab.enum_values("domain")
        assert "cognitive_behavioral" in domains
        assert "dialectical_behavior" in domains
        assert "cardiology" not in domains  # old biomedical value must be gone

    def test_array_fields_detected(self, vocab: SchemaVocabulary) -> None:
        assert vocab.is_array("session_event_tags")
        assert not vocab.is_array("domain")
        assert "therapeutic_modality" in vocab.array_fields

    def test_integer_fields_detected(self, vocab: SchemaVocabulary) -> None:
        assert vocab.integer_fields >= {
            "page_start",
            "page_end",
            "chunk_index",
            "year_published",
            "sample_size",
        }

    def test_nullability(self, vocab: SchemaVocabulary) -> None:
        assert vocab.is_nullable("practice_recommendation_level")
        assert vocab.is_nullable("year_published")
        assert not vocab.is_nullable("domain")

    def test_removed_fields_absent(self, vocab: SchemaVocabulary) -> None:
        """entities and evidence_level were removed from the schema."""
        assert "entities" not in vocab.field_names
        assert "evidence_level" not in vocab.field_names


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_enum_scalar_default_from_schema(self, vocab: SchemaVocabulary) -> None:
        assert vocab.default("session_phase") == "any"
        assert vocab.default("corpus_scope") == "rta_and_asa"

    def test_enum_scalar_without_default_prefers_other(
        self, vocab: SchemaVocabulary
    ) -> None:
        # domain declares no default but its enum includes "other".
        assert vocab.default("domain") == "other"

    def test_enum_scalar_without_default_or_other_is_empty(
        self, vocab: SchemaVocabulary
    ) -> None:
        # doc_type has no default and no "other" member.
        assert vocab.default("doc_type") == ""

    def test_array_sentinel_defaults(self, vocab: SchemaVocabulary) -> None:
        assert vocab.default("session_event_tags") == ["none"]
        assert vocab.default("patient_population") == ["not_specified"]
        assert vocab.default("keywords") == []

    def test_array_default_is_fresh_copy(self, vocab: SchemaVocabulary) -> None:
        a = vocab.default("session_event_tags")
        a.append("mutated")
        assert vocab.default("session_event_tags") == ["none"]


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_invalid_enum_falls_back_to_default(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("domain", "not_a_domain") == "other"
        assert vocab.coerce_value("doc_type", "not_a_type") == ""

    def test_valid_enum_preserved(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("domain", "interpersonal") == "interpersonal"

    def test_string_year_coerced_to_int(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("year_published", "2019") == 2019

    def test_non_numeric_year_becomes_none(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("year_published", "n/a") is None

    def test_bool_is_not_accepted_as_int(self, vocab: SchemaVocabulary) -> None:
        # bool is an int subclass; it must not slip through as sample_size.
        assert vocab.coerce_value("sample_size", True) is None

    def test_single_string_wrapped_in_array(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("keywords", "solo") == ["solo"]

    def test_array_enum_filters_invalid_items(self, vocab: SchemaVocabulary) -> None:
        result = vocab.coerce_value(
            "session_event_tags", ["shame_activation", "bogus_event"]
        )
        assert result == ["shame_activation"]

    def test_array_enum_all_invalid_uses_default(self, vocab: SchemaVocabulary) -> None:
        assert vocab.coerce_value("session_event_tags", ["bogus"]) == ["none"]

    def test_coerce_drops_unknown_keys(self, vocab: SchemaVocabulary) -> None:
        out = vocab.coerce(
            {"entities": ["x"], "evidence_level": "A", "domain": "mindfulness_based"}
        )
        assert out == {"domain": "mindfulness_based"}
