"""
Central configuration — loads all environment variables with validation.

All GCP resource IDs and API keys are read from environment variables.
Copy .env.example to .env and populate before running any pipeline component.

Usage:
    from config.settings import settings
    print(settings.gcp_project_id)
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load key=value pairs from a local .env into os.environ at import time.

    This means callers no longer have to `set -a; source .env; set +a` before
    running a script — importing `config.settings` is enough. Real environment
    variables always win (``override=False``): anything already exported takes
    precedence over the file. Set ``ENV_FILE`` to point at a different file.

    Uses python-dotenv when installed; otherwise falls back to a minimal parser
    so there is no hard dependency.
    """
    env_path = Path(os.environ.get("ENV_FILE", ".env"))
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    # Fallback: parse KEY=VALUE lines ourselves (no python-dotenv available).
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)
    except OSError:
        pass


_load_dotenv()


class Settings:
    """Validated configuration loaded from environment variables.

    Raises:
        EnvironmentError: On first access of a required variable that is missing.
    """

    # -----------------------------------------------------------------------
    # GCP Core
    # -----------------------------------------------------------------------

    @property
    def gcp_project_id(self) -> str:
        """GCP project ID. Required for all Google Cloud API calls."""
        return self._require("GCP_PROJECT_ID")

    @property
    def gcp_location(self) -> str:
        """GCP region / location (e.g. 'us', 'global', 'us-central1').
        Defaults to 'global' for Vertex AI Search."""
        return os.environ.get("GCP_LOCATION", "global")

    @property
    def gcs_datastore_region(self) -> str:
        """
        Derives the macro multi-region string required for Discovery Engine APIs
        ('us', 'eu', or 'global') from the underlying granular gcp_location.
        """
        loc = self.gcp_location.strip().lower()    
        # United States prefixes
        if loc.startswith("us"):
            return "us"
        # European prefixes (safely catches both macro 'eu' and regional 'europe-west1')
        if loc.startswith(("eu", "europe")):
            return "eu"
        # Fallback for explicit 'global' or unexpected local regions
        return "global"

    @property
    def discovery_engine_location(self) -> str:
        """Discovery Engine multi-region ('global' | 'us' | 'eu').

        Discovery Engine (Vertex AI Search) only accepts these three locations —
        NOT compute regions like 'us-central1'. This derives the correct value
        from ``gcp_location`` (e.g. 'us-central1' -> 'us'). **Route every
        Discovery Engine call through this, never the raw ``gcp_location``**, or
        the request is sent to a region the endpoint cannot serve
        (INVALID_ARGUMENT). This is exactly what crashed the first ingest.
        """
        return self.gcs_datastore_region

    @property
    def discovery_engine_endpoint(self) -> str | None:
        """Regional API endpoint for the Discovery Engine client, or None (global).

        For a non-global location the client MUST target
        ``{region}-discoveryengine.googleapis.com`` or requests are misrouted and
        either fail with INVALID_ARGUMENT or hang until timeout. Pass this to a
        client via ``ClientOptions(api_endpoint=...)``.
        """
        loc = self.discovery_engine_location
        if loc == "global":
            return None
        return f"{loc}-discoveryengine.googleapis.com"

    # -----------------------------------------------------------------------
    # Google Cloud Storage
    # -----------------------------------------------------------------------

    @property
    def gcs_bucket_name(self) -> str:
        """GCS bucket for staging PDFs and chunk JSONL files."""
        return self._require("GCS_BUCKET_NAME")

    # -----------------------------------------------------------------------
    # Vertex AI Search (Discovery Engine)
    # -----------------------------------------------------------------------

    @property
    def vertex_search_datastore_id(self) -> str:
        """Discovery Engine DataStore resource ID (not full resource name)."""
        return self._require("VERTEX_SEARCH_DATASTORE_ID")

    @property
    def vertex_search_engine_id(self) -> str:
        """Discovery Engine Search Engine resource ID."""
        return self._require("VERTEX_SEARCH_ENGINE_ID")

    # -----------------------------------------------------------------------
    # Gemini / Vertex AI
    # -----------------------------------------------------------------------

    @property
    def gemini_api_key(self) -> str | None:
        """DEPRECATED (2026-07-27). The pipeline uses Vertex AI + ADC, not an API
        key — see devlog.md#DONE(genai-migration). Retained only so any external
        tooling that still reads it does not break; nothing in the pipeline uses it."""
        return os.environ.get("GEMINI_API_KEY")

    @property
    def gemini_model_metadata(self) -> str:
        """Gemini model for metadata generation. Defaults to gemini-1.5-pro."""
        return os.environ.get("GEMINI_MODEL_METADATA", "gemini-1.5-pro")

    @property
    def gemini_model_generation(self) -> str:
        """Gemini model for response generation. Defaults to gemini-1.5-pro."""
        return os.environ.get("GEMINI_MODEL_GENERATION", "gemini-1.5-pro")

    @property
    def gemini_model_reranker(self) -> str:
        """Gemini model for re-ranking. Defaults to gemini-1.5-flash (lower cost)."""
        return os.environ.get("GEMINI_MODEL_RERANKER", "gemini-1.5-flash")

    # -----------------------------------------------------------------------
    # Google Document AI (optional — OCR fallback)
    # -----------------------------------------------------------------------

    @property
    def document_ai_processor_id(self) -> str | None:
        """Full Document AI processor resource name. None disables OCR fallback."""
        return os.environ.get("DOCUMENT_AI_PROCESSOR_ID")

    @property
    def use_document_ai(self) -> bool:
        """If True, always use Document AI instead of pdfplumber."""
        return os.environ.get("USE_DOCUMENT_AI", "false").lower() == "true"

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @property
    def corpus_dir(self) -> Path:
        """Local directory to watch for incoming PDFs."""
        return Path(os.environ.get("CORPUS_DIR", "corpus"))

    @property
    def manifest_path(self) -> Path:
        """Path to the ingestion manifest JSON file."""
        return Path(os.environ.get("MANIFEST_PATH", ".ingestion_manifest.json"))

    @property
    def ingest_strategy(self) -> str:
        """Chunk-processing strategy name ('chapter' | 'independent').

        'chapter' (default) tags each chunk with its chapter context and runs
        chapters concurrently but chunks-within-a-chapter sequentially.
        'independent' tags every chunk in isolation at maximum concurrency.
        See ingestion/processing_strategy.py.
        """
        return os.environ.get("INGEST_STRATEGY", "chapter")

    @property
    def ingest_concurrency(self) -> int:
        """Max units (chapters / chunks) tagged concurrently. Keep modest to
        respect the Gemini/Vertex quota. Default 4."""
        try:
            return max(1, int(os.environ.get("INGEST_CONCURRENCY", "4")))
        except ValueError:
            return 4

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for per-document metadata checkpoints (crash-safe resume).

        batch_ingest writes each chunk's generated metadata here as it goes, so a
        crash preserves the expensive Gemini output and a re-run resumes instead
        of re-paying. One JSONL per doc_id.
        """
        return Path(os.environ.get("CHECKPOINT_DIR", "ingestion_checkpoints"))

    @property
    def chunk_config_path(self) -> Path:
        """Path to chunk_config.yaml."""
        return Path(os.environ.get("CHUNK_CONFIG_PATH", "config/chunk_config.yaml"))

    @property
    def prompt_config_path(self) -> Path:
        """Path to prompt_config.yaml (query-time persona templates)."""
        return Path(os.environ.get("PROMPT_CONFIG_PATH", "config/prompt_config.yaml"))

    @property
    def ingestion_prompt_path(self) -> Path:
        """Path to ingestion_prompt.yaml (hand-authored Gemini extraction framing).

        Holds the natural-language scaffolding/guidance for metadata_gen's prompt;
        the enum vocabulary is generated from the schema, not stored here. Missing
        or invalid file → metadata_gen uses built-in defaults.
        """
        return Path(os.environ.get("INGESTION_PROMPT_PATH", "config/ingestion_prompt.yaml"))

    @property
    def metadata_schema_path(self) -> Path:
        """Path to the ACTIVE ingestion schema.

        Defaults to ``config/rta_v1.json`` — the real-time-analysis (RTA) schema
        (``ChunkMetadata_v1``, 23 fields) that ``metadata_gen`` and
        ``setup_vertex_search`` load. The 37-field unified schema in
        ``config/metadata_schema.json`` is retained for future ASA work and as a
        historical record; point ``METADATA_SCHEMA_PATH`` at it only for ASA.
        """
        return Path(
            os.environ.get("METADATA_SCHEMA_PATH", "config/rta_v1.json")
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _require(self, key: str) -> str:
        """Return the value of a required environment variable.

        Args:
            key: Environment variable name.

        Returns:
            Non-empty string value.

        Raises:
            EnvironmentError: If the variable is not set or is empty.
        """
        value = os.environ.get(key, "").strip()
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set. "
                f"Copy .env.example to .env and populate it."
            )
        return value

    def validate_all(self) -> None:
        """Eagerly validate all required settings.

        Call this at application startup to fail fast rather than discovering
        missing variables mid-pipeline.

        Raises:
            EnvironmentError: On the first missing required variable.
        """
        _ = self.gcp_project_id
        _ = self.gcs_bucket_name
        _ = self.vertex_search_datastore_id
        _ = self.vertex_search_engine_id


settings = Settings()
