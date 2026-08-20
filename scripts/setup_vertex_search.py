"""
One-time provisioning script for Vertex AI Search resources.

Run this ONCE before any ingestion. Creates:
  1. An unstructured DataStore (enterprise edition for semantic search)
  2. Registers the ChunkMetadata schema with filterable attributes
  3. Creates a Search Engine backed by the DataStore

CAUTION:
  - DataStore region is immutable after creation. Confirm GCP_LOCATION before running.
  - Schema changes after ingestion require a full re-import (see issues.md P3).
  - This script is IDEMPOTENT: safe to re-run; existing resources are detected
    and skipped rather than recreated.

Usage:
    python scripts/setup_vertex_search.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from google.api_core import retry as api_retry
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, RetryError, AlreadyExists, DeadlineExceeded, ServiceUnavailable
from google.cloud import discoveryengine_v1 as discoveryengine

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_LRO_RPC_TIMEOUT = 300  # seconds for the initial RPC that starts a long-running operation
_LRO_RESULT_TIMEOUT = 600  # seconds to wait for the LRO (whole process) to complete -> 300 (5 minutes) increased to 600 (10 minutes)
_RETRY_TRANSIENT = api_retry.Retry(
    predicate=api_retry.if_exception_type(DeadlineExceeded, ServiceUnavailable),
    initial=2.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=300.0, # retry extended 120 -> 300 in case GCP scale up
)


def _client_options() -> ClientOptions | None:
    """Resolve the regional API endpoint for settings.gcp_location.

    The Discovery Engine client libraries default to the *global* endpoint
    (discoveryengine.googleapis.com) regardless of settings.gcp_location. If
    GCP_LOCATION is a non-global region (e.g. "us", "eu"), calls made against
    the global endpoint for a regional parent resource are misrouted: they
    don't fail fast, they hang until the RPC/LRO timeout elapses with no
    useful error — a "gRPC sinkhole". Passing the matching regional endpoint
    via client_options avoids this.

    Returns:
        ClientOptions with the regional api_endpoint set, or None for the
        default (global) endpoint.
    """
    endpoint = settings.discovery_engine_endpoint
    return ClientOptions(api_endpoint=endpoint) if endpoint else None


def create_datastore(dry_run: bool = False) -> str:
    """Create the unstructured DataStore for chunk documents.

    Uses enterprise_edition=True to enable semantic (vector) search.
    FULL content level stores the chunk text for snippet extraction.

    Args:
        dry_run: If True, log what would be created without calling the API.

    Returns:
        Full DataStore resource name string.

    Raises:
        google.api_core.exceptions.GoogleAPIError: On API failure.
    """
    parent = (
        f"projects/{settings.gcp_project_id}/locations/{settings.gcs_datastore_region}"
        f"/collections/default_collection"
    )
    datastore_name = f"{parent}/dataStores/{settings.vertex_search_datastore_id}"

    if dry_run:
        logger.info(
            "[DRY RUN] Would create DataStore: %s", settings.vertex_search_datastore_id
        )
        return datastore_name

    client = discoveryengine.DataStoreServiceClient(client_options=_client_options())

    datastore = discoveryengine.DataStore(
        display_name=settings.vertex_search_datastore_id,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
        solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
    )

    try:
        operation = client.create_data_store(
            parent=parent,
            data_store=datastore,
            data_store_id=settings.vertex_search_datastore_id,
            timeout=_LRO_RPC_TIMEOUT,
            retry=_RETRY_TRANSIENT,
        )
        result = operation.result(timeout=_LRO_RESULT_TIMEOUT)
        logger.info("DataStore created: %s", result.name)
        return result.name
    except AlreadyExists:
        logger.info("DataStore already exists: %s", datastore_name)
        return datastore_name


def register_schema(datastore_name: str, dry_run: bool = False) -> None:
    """Update the DataStore schema to register ChunkMetadata fields as filterable.

    Loads config/metadata_schema.json and converts it into a Discovery Engine
    `json_schema` via _create_discoveryengine_schema(): every field is annotated
    with Vertex AI Search indexing keywords (`retrievable`/`indexable`/
    `searchable`) and nullable union types (`["string", "null"]`) are collapsed
    to the single concrete type Discovery Engine requires. `indexable` is what
    makes a scalar or array field usable in filter expressions (AIP-160) and
    facets; string leaves are also `searchable` for full-text. Array fields are
    annotated on their element (`items`) leaf, so all 16 array metadata fields
    become filterable.

    NOTE: indexing is configured *inside* the JSON schema, not via a separate
    `discoveryengine.FieldConfig` object. `Schema.field_configs` is OUTPUT_ONLY
    in every API surface (v1/v1beta/v1alpha) — see discovery_engine_comparison.md.

    Must be called BEFORE the first ImportDocuments run. Calling this after
    ingestion does not retroactively index existing documents.

    Args:
        datastore_name: Full DataStore resource name from create_datastore().
        dry_run: If True, log the schema diff without updating.

    Raises:
        FileNotFoundError: If config/metadata_schema.json does not exist.
    """
    schema_data = _load_metadata_schema()

    if dry_run:
        logger.info(
            "[DRY RUN] Would register schema with %d field(s)",
            len(schema_data.get("properties", {})),
        )
        return

    client = discoveryengine.SchemaServiceClient(client_options=_client_options())
    schema_name = f"{datastore_name}/schemas/default_schema"

    # Indexing is registered by embedding Vertex AI Search keywords into the
    # JSON schema document itself (see _create_discoveryengine_schema), NOT via
    # discoveryengine.FieldConfig objects: `Schema.field_configs` is OUTPUT_ONLY
    # in v1/v1beta/v1alpha, and the old code's FieldConfig(filterable=...) block
    # referenced an API that does not exist. This conversion also covers the
    # 16 array fields (annotated on their `items` leaf) that the old
    # continue-past-array-fields loop left unregistered. See DISCREPANCIES.md and
    # discovery_engine_comparison.md.
    de_schema = _create_discoveryengine_schema(schema_data)

    schema = discoveryengine.Schema(
        name=schema_name,
        json_schema=json.dumps(de_schema),
    )

    try:
        logger.info("Providing schema to Discovery Engine: %s", schema_name)
        logger.info("Schema: %s", de_schema)
        operation = client.update_schema(
            request={"schema": schema},
            timeout=_LRO_RPC_TIMEOUT,
            retry=_RETRY_TRANSIENT,
        )
        response = operation.result(timeout=_LRO_RPC_TIMEOUT)
        logger.info("Schema successfully loaded: %s", response.name)
    except AlreadyExists: # for other errors should raise through
        logger.info("Schema already exists, no changes made: %s", schema_name)


def create_search_engine(datastore_name: str, dry_run: bool = False) -> str:
    """Create a Search Engine backed by the DataStore.

    Configures the engine for:
      - GENERIC vertical (suitable for unstructured documents)
      - search_tier = STANDARD (upgrade to ENTERPRISE for advanced features)
      - Hybrid search (semantic + keyword) via serving config

    Args:
        datastore_name: Full DataStore resource name.
        dry_run: If True, log what would be created without calling the API.

    Returns:
        Full Search Engine resource name string.
    """
    parent = (
        f"projects/{settings.gcp_project_id}/locations/{settings.gcs_datastore_region}"
        f"/collections/default_collection"
    )
    engine_name = f"{parent}/engines/{settings.vertex_search_engine_id}"

    if dry_run:
        logger.info(
            "[DRY RUN] Would create Search Engine: %s", settings.vertex_search_engine_id
        )
        return engine_name

    client = discoveryengine.EngineServiceClient(client_options=_client_options())

    engine = discoveryengine.Engine(
        display_name=settings.vertex_search_engine_id,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        data_store_ids=[settings.vertex_search_datastore_id],
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_STANDARD,
        ),
    )

    try:
        operation = client.create_engine(
            parent=parent,
            engine=engine,
            engine_id=settings.vertex_search_engine_id,
            timeout=_LRO_RPC_TIMEOUT,
            retry=_RETRY_TRANSIENT,
        )
        result = operation.result(timeout=_LRO_RESULT_TIMEOUT)
        logger.info("Search Engine created: %s", result.name)
        return result.name
    except AlreadyExists:
        logger.info("Search Engine already exists: %s", engine_name)
        return engine_name


# Fields stored and returned but intentionally NOT registered as filterable.
# Per config/metadata_schema.json notes.vertex_ai_search, `missingness` is
# informational only and does not require filterable registration.
_INFORMATIONAL_ONLY_FIELDS = frozenset({"missingness"})

# JSON-Schema draft used by Discovery Engine custom schemas.
_DISCOVERYENGINE_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_JSON_NULL = "null"


def _concrete_type(json_type: object) -> object:
    """Collapse a JSON-Schema `type` declaration to a single Discovery Engine type.

    `config/metadata_schema.json` marks optional scalars with a *union* type such
    as ``["string", "null"]`` / ``["integer", "null"]`` — JSON-Schema's idiom for
    "nullable". Discovery Engine's schema requires `type` to be a **single**
    string (one of string/number/integer/boolean/array/object/datetime/
    geolocation); a JSON list is not an accepted value and the union carries no
    indexing meaning. This returns the first non-null member, so
    ``["integer", "null"]`` -> ``"integer"`` and ``"string"`` passes through
    unchanged.

    Args:
        json_type: The `type` value from a schema property or items node.

    Returns:
        The concrete single type string, or the input unchanged if it is not a
        union list.
    """
    if isinstance(json_type, list):
        non_null = [t for t in json_type if t != _JSON_NULL]
        return non_null[0] if non_null else None
    return json_type


def _create_discoveryengine_schema(schema_data: dict) -> dict:
    """Convert `config/metadata_schema.json` into a Discovery Engine `json_schema`.

    `metadata_schema.json` is authored as a JSON-Schema *validation* document
    (enums, `minimum`/`maximum`, `pattern`, nullable union types, custom
    `notes`). Discovery Engine wants a *field-configuration* document: the same
    property tree, but with per-field indexing keywords and a single concrete
    `type` per field. This function performs that translation without mutating
    the source schema.

    Two conversions happen here:

    1. **Indexing keywords.** Every field's leaf node is annotated with the
       Vertex AI Search booleans — `retrievable`, `indexable`, `searchable` —
       because `Schema.field_configs` is OUTPUT_ONLY on every API surface and
       indexing must be expressed *inside* the schema document (see
       discovery_engine_comparison.md and CHANGELOG 2026-07-10). Keyword
       semantics:
         - retrievable: field is returned on the SearchResult document. Required
           to rebuild ChunkMetadata and to construct page-level citations.
         - indexable: field is usable in AIP-160 filter expressions and facets —
           the "filterable" requirement in notes.vertex_ai_search. Set on every
           field except the informational-only `missingness`.
         - searchable: field contributes to full-text search. Set only on string
           leaves (not integer/number/boolean).

       For `type: array` fields the keywords attach to the element (`items`)
       leaf — this is the placement Discovery Engine documents and expects for
       arrays of primitives, and is what makes all 16 array metadata fields
       (therapeutic_modality, session_event_tags, clinical_presentation,
       risk_dimension_tags, ...) individually filterable. Note this deliberately
       contradicts schema_notes.md, whose "Strict Array Constraint" (flags at the
       property level, never inside `items`) is inverted relative to Google's
       live documentation; following it would silently leave every array field
       unregistered and break the RTA event filter. See metadata_summary.md and
       architecture_bootstrap.md.

    2. **Nullable union collapse.** Optional scalars declared as
       ``["string", "null"]`` / ``["integer", "null"]`` are collapsed to their
       concrete type via `_concrete_type`, because Discovery Engine rejects a
       list-valued `type`. This affects year_published, sample_size,
       practice_recommendation_level, training_level_required, and study_type.

    The returned document keeps each property's `enum`/`description` (harmless to
    Discovery Engine, useful for humans) but drops the source-schema top-level
    scaffolding that is not part of a field-config schema (`title`,
    `description`, custom `notes`, `additionalProperties`).

    Args:
        schema_data: Parsed metadata_schema.json (not mutated).

    Returns:
        A Discovery Engine-ready schema dict suitable for `Schema.json_schema`.
    """
    properties_out: dict = {}
    for field_name, prop in schema_data.get("properties", {}).items():
        prop = copy.deepcopy(prop)
        is_array = prop.get("type") == "array"
        # Collapse any nullable union on the property itself first.
        prop["type"] = _concrete_type(prop.get("type"))

        # For arrays the keywords attach to the element schema, not the array node.
        if is_array:
            leaf = prop.setdefault("items", {})
            leaf["type"] = _concrete_type(leaf.get("type"))
        else:
            leaf = prop

        leaf["retrievable"] = True
        if field_name not in _INFORMATIONAL_ONLY_FIELDS:
            leaf["indexable"] = True
            if leaf.get("type") == "string":
                leaf["searchable"] = True

        properties_out[field_name] = prop

    return {
        "$schema": _DISCOVERYENGINE_SCHEMA_DRAFT,
        "type": "object",
        "properties": properties_out,
        "required": list(schema_data.get("required", [])),
    }


def _load_metadata_schema() -> dict:
    """Load and return the parsed metadata_schema.json.

    Returns:
        Parsed schema dict.

    Raises:
        FileNotFoundError: If the schema file is missing.
    """
    schema_path = settings.metadata_schema_path
    if not schema_path.exists():
        raise FileNotFoundError(f"Metadata schema not found: {schema_path}")
    with open(schema_path) as f:
        return json.load(f)


def main() -> None:
    """Entry point: parse arguments and run provisioning in order."""
    parser = argparse.ArgumentParser(
        description="Provision Vertex AI Search resources for the RAG pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be created without making API calls.",
    )
    args = parser.parse_args()

    logger.info("Validating settings...")
    settings.validate_all()

    logger.info("Step 1/3: Creating DataStore...")
    datastore_name = create_datastore(dry_run=args.dry_run)

    logger.info("Step 2/3: Registering schema...")
    register_schema(datastore_name, dry_run=args.dry_run)

    logger.info("Step 3/3: Creating Search Engine...")
    engine_name = create_search_engine(datastore_name, dry_run=args.dry_run)

    logger.info("Done. Resources:")
    logger.info("  DataStore: %s", datastore_name)
    logger.info("  Engine:    %s", engine_name)
    if args.dry_run:
        logger.info("(dry-run: no resources were created)")


if __name__ == "__main__":
    main()
