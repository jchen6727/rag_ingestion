"""
Schema-driven controlled-vocabulary loader for the ingestion pipeline.

`config/metadata_schema.json` is the single authoritative source for the
psychotherapy metadata vocabulary (unified RTA + ASA schema). Historically the
controlled vocabulary was *also* hard-coded inside `ingestion/metadata_gen.py`
(`_VALID_DOMAINS`, `_VALID_DOC_TYPES`, `_VALID_EVIDENCE_LEVELS`) as an old
biomedical enum set, which silently drifted from the schema. See DISCREPANCIES.md
("Metadata schema vs code").

This module removes that second source of truth. `SchemaVocabulary` parses the
JSON schema once and exposes its enums, array-field membership, nullability,
defaults, and required-field list as a structured object. Both
`ingestion/metadata_gen.py` (coercion + prompt construction) and
`scripts/setup_vertex_search.py` (DataStore field registration) consume it so
that the vocabulary is derived from `metadata_schema.json` in exactly one place.

Usage:
    from config.schema_loader import SchemaVocabulary

    vocab = SchemaVocabulary.from_path()          # default config path
    vocab.enum_values("domain")                   # -> {"cognitive_behavioral", ...}
    vocab.is_array("session_event_tags")          # -> True
    clean = vocab.coerce(raw_gemini_dict)          # schema-validated dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Sentinel distinguishing "schema declares no default" from "default is null".
_NO_DEFAULT = object()


class SchemaVocabulary:
    """Structured view over ``config/metadata_schema.json``.

    Wraps the parsed JSON-Schema ``properties`` block and answers the questions
    the ingestion pipeline needs to enforce the controlled vocabulary:

      - What are the valid enum values for a field?
      - Is a field an array, an integer, or nullable?
      - What default should an absent or invalid value fall back to?
      - Which fields are required / arrays / integer-typed (for DataStore
        registration)?

    The class holds no I/O state; construct it from an already-parsed schema
    dict, or use :meth:`from_path` to load and parse the JSON file.
    """

    def __init__(self, schema: dict) -> None:
        """Args:
        schema: Parsed ``metadata_schema.json`` object (must contain a
            top-level ``properties`` mapping; ``required`` is optional).
        """
        self._schema = schema
        self.properties: dict[str, dict] = schema.get("properties", {})
        self.required: list[str] = list(schema.get("required", []))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, schema_path: Optional[Path] = None) -> "SchemaVocabulary":
        """Load and parse the metadata schema from disk.

        Args:
            schema_path: Explicit path to ``metadata_schema.json``. If ``None``,
                resolves to the project-local ``config/metadata_schema.json``
                (the same file `MetadataGenerator` loads by default).

        Returns:
            A ``SchemaVocabulary`` built from the parsed schema.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "metadata_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Metadata schema not found: {schema_path}")
        with open(schema_path) as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    # Field introspection
    # ------------------------------------------------------------------

    @property
    def field_names(self) -> list[str]:
        """All field names declared in the schema, in schema order."""
        return list(self.properties.keys())

    def _types(self, field: str) -> list[str]:
        """Return the declared JSON type(s) of ``field`` as a list."""
        t = self.properties.get(field, {}).get("type")
        if t is None:
            return []
        return [t] if isinstance(t, str) else list(t)

    def is_array(self, field: str) -> bool:
        """True if ``field`` is declared as a JSON array."""
        return "array" in self._types(field)

    def is_integer(self, field: str) -> bool:
        """True if ``field`` accepts a JSON integer (nullable or not)."""
        return "integer" in self._types(field)

    def is_nullable(self, field: str) -> bool:
        """True if ``field`` accepts null (via its ``type`` list or a null enum member)."""
        if "null" in self._types(field):
            return True
        enum = self.properties.get(field, {}).get("enum")
        return isinstance(enum, list) and None in enum

    def enum_values(self, field: str) -> Optional[set[str]]:
        """Return the non-null enum members for ``field``, or ``None`` if unconstrained.

        For array fields the item-level enum is returned. ``None`` means the
        field has no enum (free-text scalar or free-text array such as
        ``keywords`` / ``technique_tags``).
        """
        prop = self.properties.get(field, {})
        if self.is_array(field):
            enum = prop.get("items", {}).get("enum")
        else:
            enum = prop.get("enum")
        if not isinstance(enum, list):
            return None
        return {v for v in enum if v is not None}

    def default(self, field: str) -> Any:
        """Return the schema-declared default for ``field``.

        Falls back to a type-appropriate empty value when the schema declares no
        default: ``[]`` for arrays, ``None`` for nullable/integer scalars,
        ``"other"`` for enum scalars that include it, else ``""``.
        """
        prop = self.properties.get(field, {})
        declared = prop.get("default", _NO_DEFAULT)
        if declared is not _NO_DEFAULT:
            return list(declared) if isinstance(declared, list) else declared
        if self.is_array(field):
            return []
        enum = self.enum_values(field)
        if enum and "other" in enum:
            return "other"
        if self.is_nullable(field) or self.is_integer(field):
            return None
        return ""

    # ------------------------------------------------------------------
    # Field-set accessors (used by setup_vertex_search.py)
    # ------------------------------------------------------------------

    @property
    def array_fields(self) -> set[str]:
        """Names of all array-typed fields."""
        return {f for f in self.properties if self.is_array(f)}

    @property
    def integer_fields(self) -> set[str]:
        """Names of all integer-typed (possibly nullable) fields."""
        return {f for f in self.properties if self.is_integer(f)}

    @property
    def scalar_enum_fields(self) -> set[str]:
        """Names of scalar (non-array) fields constrained by an enum."""
        return {
            f
            for f in self.properties
            if not self.is_array(f) and self.enum_values(f) is not None
        }

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    def coerce(self, raw: dict) -> dict:
        """Coerce a raw (e.g. Gemini) response dict to schema-valid values.

        For every key that exists in the schema, the value is normalized to a
        schema-conformant value (enum membership enforced, arrays normalized,
        integers parsed, invalid values replaced by defaults). Keys **not** in
        the schema — such as the removed ``entities`` / ``evidence_level``
        fields — are dropped, matching the schema's ``additionalProperties:
        false`` contract.

        Provenance fields (``doc_id``, ``page_start``, ``page_end``,
        ``chunk_index``) are coerced like any other field here; callers are
        responsible for overriding them from the source ``Chunk`` afterward.

        Args:
            raw: Parsed model output.

        Returns:
            A new dict containing only schema-recognized keys with coerced values.
        """
        out: dict[str, Any] = {}
        for field, value in raw.items():
            if field not in self.properties:
                continue
            out[field] = self.coerce_value(field, value)
        return out

    def coerce_value(self, field: str, value: Any) -> Any:
        """Coerce a single field's value to a schema-conformant value.

        Args:
            field: Schema field name (must exist in ``properties``).
            value: Raw value to coerce.

        Returns:
            The coerced value (enum-checked scalar, normalized array, parsed
            integer, or the field's default when the input is invalid).
        """
        if self.is_array(field):
            return self._coerce_array(field, value)
        if self.is_integer(field):
            return self._coerce_int(field, value)
        enum = self.enum_values(field)
        if enum is not None:
            return self._coerce_enum(field, value, enum)
        # Free-text scalar (doc_id, source_file, title, subdomain, chapter).
        return value if isinstance(value, str) else self.default(field)

    def _coerce_enum(self, field: str, value: Any, enum: set[str]) -> Any:
        if value in enum:
            return value
        if value is None and self.is_nullable(field):
            return None
        return self.default(field)

    def _coerce_int(self, field: str, value: Any) -> Any:
        default = self.default(field)
        if isinstance(value, bool):  # bool is an int subclass; reject it
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except (ValueError, TypeError):
                return default
        return default

    def _coerce_array(self, field: str, value: Any) -> list:
        default = self.default(field)  # already a fresh list copy
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            return default
        item_enum = self.enum_values(field)
        if item_enum is not None:
            cleaned = [v for v in value if v in item_enum]
        else:
            cleaned = [v for v in value if isinstance(v, str)]
        return cleaned if cleaned else default


def load_vocabulary(schema_path: Optional[Path] = None) -> SchemaVocabulary:
    """Convenience wrapper for :meth:`SchemaVocabulary.from_path`."""
    return SchemaVocabulary.from_path(schema_path)
