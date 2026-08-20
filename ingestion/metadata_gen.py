"""
AI-guided metadata generation using Google Gemini.

Sends each chunk (plus a configurable surrounding context window) to Gemini
with a structured extraction prompt. The response is validated against the
ChunkMetadata Pydantic schema. On parse failure, falls back to rule-based
heuristics so that ingestion never hard-fails due to a bad Gemini response.

See caveats.md §2 (Gemini output reliability) and issues.md P3 (schema drift).

Rate limiting: standard Gemini 1.5 Pro quotas are ~360 RPM. For large batches
use exponential backoff or switch to Vertex AI Batch Prediction.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from config.schema_loader import SchemaVocabulary
from config.settings import settings
from models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# Default retry parameters for Gemini API calls
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled each attempt

# The controlled vocabulary (valid domains, doc_types, enums, array fields, and
# defaults) is NOT hard-coded here — it is derived from config/metadata_schema.json
# at construction time via SchemaVocabulary, so metadata_gen can never drift from
# the authoritative schema. See config/schema_loader.py and DISCREPANCIES.md.

# Provenance fields are always taken from the source Chunk, never trusted from
# Gemini output.
_PROVENANCE_FIELDS = ("doc_id", "page_start", "page_end", "chunk_index")


class MetadataGenerator:
    """
    Calls Gemini to extract structured metadata from document chunks.

    Usage:
        gen = MetadataGenerator(model_name="gemini-1.5-pro")
        metadata = gen.generate(chunk, context_window=preceding_text)
        chunk.metadata = metadata
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        schema_path: Optional[Path] = None,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model_name: Gemini model identifier. gemini-1.5-pro is recommended
                        for long-context metadata extraction; gemini-1.5-flash
                        is acceptable for short chunks if cost is a concern.
            schema_path: Path to the active metadata schema (config/rta_v1.json).
                         Loaded at init; defaults to the project-local path if None.
            temperature: Gemini sampling temperature. 0.0 for deterministic output.

        Auth: uses Vertex AI via Application Default Credentials (ADC) — there is
        no API-key path (migrated 2026-07-27, devlog.md#DONE(genai-migration)).
        Run ``gcloud auth application-default login`` and set GCP_PROJECT_ID.
        """
        self._model_name = model_name
        self._temperature = temperature
        self._schema = self._load_schema(schema_path)
        # Controlled vocabulary derived from the authoritative schema; drives
        # both prompt construction and response coercion.
        self._vocab = SchemaVocabulary(self._schema)
        # Hand-authored prompt scaffolding (config/ingestion_prompt.yaml); falls
        # back to built-in defaults if the file is missing or unreadable.
        self._prompt_cfg = self._load_prompt_config()
        self._client: Optional[genai.Client] = None

    def generate(self, chunk: Chunk, context_window: str = "") -> ChunkMetadata:
        """Generate metadata for a single chunk.

        Builds the extraction prompt, calls Gemini with retry/backoff, validates
        the response, and returns a ChunkMetadata. Falls back to rule-based
        extraction if Gemini fails after all retries.

        Args:
            chunk: The chunk to generate metadata for.
            context_window: Optional preceding text (e.g., previous chunk or section
                            header) to give Gemini additional context.

        Returns:
            Validated ChunkMetadata. Fields that cannot be inferred are set to
            their default values (empty string / None / empty list).
        """
        try:
            prompt = self._build_extraction_prompt(chunk, context_window)
            raw = self._call_gemini(prompt)
            return self._validate_and_coerce(raw, chunk)
        except Exception as exc:
            logger.warning(
                "Metadata generation failed for chunk %s: %s — using fallback",
                chunk.chunk_id,
                exc,
            )
            return self._fallback_extraction(chunk)

    def generate_batch(
        self,
        chunks: list[Chunk],
        context_window: str = "",
        delay_between_calls: float = 0.1,
    ) -> list[ChunkMetadata]:
        """Generate metadata for a list of chunks sequentially with rate-limiting.

        For large batches (>1000 chunks), consider switching to Vertex AI Batch
        Prediction to avoid hitting online quota limits.

        Args:
            chunks: Chunks to process, in order.
            context_window: Shared context text passed to every generate() call.
            delay_between_calls: Seconds to sleep between calls for rate limiting.

        Returns:
            List of ChunkMetadata in the same order as the input chunks.
        """
        results: list[ChunkMetadata] = []
        for chunk in chunks:
            metadata = self.generate(chunk, context_window)
            results.append(metadata)
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)
        return results

    def _build_extraction_prompt(self, chunk: Chunk, context_window: str) -> str:
        """Build the Gemini extraction prompt for a chunk.

        The prompt includes:
          - The JSON schema of output fields (from self._schema)
          - An inline enum legend listing the allowed values per controlled field
            (derived from the schema via SchemaVocabulary)
          - Psychotherapy-specific extraction guidance from the schema `notes`
            (domain-vs-modality distinction, missingness inference, patient-facing
            exclusion) so RTA/ASA fields are elicited correctly
          - The context window (if provided)
          - The chunk text
          - Explicit instruction to output valid JSON only

        Args:
            chunk: Chunk to extract metadata for.
            context_window: Preceding text for additional context.

        Returns:
            Full prompt string ready to send to Gemini.
        """
        properties = self._schema.get("properties", {})
        schema_excerpt = json.dumps(properties, indent=2)
        cfg = self._prompt_cfg

        parts = [
            cfg["system_preamble"],
            cfg["output_instruction"],
            "",
            schema_excerpt,
            "",
            cfg["allowed_values_header"],
            self._enum_legend(),
            "",
            self._extraction_guidance(),
            "",
        ]

        if context_window:
            parts += ["Context (preceding text):", context_window, ""]

        parts += [
            "Chunk text to analyze:",
            chunk.text,
            "",
            cfg["closing_instruction"],
        ]

        return "\n".join(parts)

    def _enum_legend(self) -> str:
        """Build an inline "field: allowed values" legend from the schema enums."""
        lines: list[str] = []
        for field in self._vocab.field_names:
            enum = self._vocab.enum_values(field)
            if not enum:
                continue
            marker = " (array; select all that apply)" if self._vocab.is_array(field) else ""
            allowed = ", ".join(sorted(enum))
            if self._vocab.is_nullable(field):
                allowed += ", null"
            lines.append(f"- {field}{marker}: {allowed}")
        return "\n".join(lines)

    def _extraction_guidance(self) -> str:
        """Return the clinical extraction guidance block for the prompt.

        Base guidance bullets come from ``config/ingestion_prompt.yaml`` (or the
        built-in defaults); the schema ``notes`` (``domain_vs_modality`` and
        ``directionality_applies_when``) are appended so the guidance always
        tracks the active schema.
        """
        notes = self._schema.get("notes", {})
        lines: list[str] = ["Extraction guidance:"]
        lines.extend(self._prompt_cfg.get("guidance", []))
        for key in ("domain_vs_modality", "directionality_applies_when"):
            if notes.get(key):
                lines.append("- " + notes[key])
        return "\n".join(line for line in lines if str(line).strip())

    # Built-in fallback used when config/ingestion_prompt.yaml is absent/invalid.
    _DEFAULT_PROMPT_CFG = {
        "system_preamble": (
            "You are extracting structured metadata for a psychotherapy guidance "
            "RAG corpus (CBT/DBT/IPT clinicians; real-time RTA retrieval pipeline)."
        ),
        "output_instruction": "Respond ONLY with valid JSON matching this schema (no markdown, no prose):",
        "allowed_values_header": "Allowed values for controlled fields (use these EXACT tokens):",
        "guidance": [
            "- domain is document-level (the source's overall orientation); therapeutic_modality "
            "is chunk-level (what THIS passage addresses) — they may differ.",
            "- directionality and applies_when are ONE decision: does the passage say to DO something "
            "(indicated), AVOID something (contraindicated), or proceed with CAUTION (cautionary), and "
            "under exactly which events/presentations/patient-states does that apply? If it applies "
            "whenever the modality is active, return applies_when as an empty list. A contraindicated "
            "or cautionary passage MUST list at least one applies_when value.",
            "- clinical_caution and any contraindicated/cautionary directionality are patient-safety "
            "fields: extract them explicitly and never omit a stated contraindication.",
            "- session_event_tags here tag what the passage is ABOUT (a retrieval target), NOT a live "
            "event; 'none' means the passage addresses no specific in-session event.",
        ],
        "closing_instruction": "Output valid JSON only.",
    }

    def _load_prompt_config(self) -> dict:
        """Load config/ingestion_prompt.yaml, merged over the built-in defaults.

        Any key the file omits (or the whole file, if missing/invalid) falls back
        to ``_DEFAULT_PROMPT_CFG`` so a bad edit degrades gracefully rather than
        breaking ingestion.
        """
        cfg = dict(self._DEFAULT_PROMPT_CFG)
        try:
            import yaml  # local import: metadata_gen has no hard yaml dependency otherwise

            from config.settings import settings

            path = settings.ingestion_prompt_path
            if path.exists():
                loaded = yaml.safe_load(path.read_text()) or {}
                if isinstance(loaded, dict):
                    cfg.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:  # noqa: BLE001 — never fail ingestion on prompt config
            logger.warning("Could not load ingestion prompt config (%s); using defaults.", exc)
        return cfg

    def _call_gemini(self, prompt: str) -> dict:
        """Send prompt to Gemini and return the parsed JSON response.

        Uses response_mime_type="application/json" to request structured output.
        Retries on transient errors with exponential backoff.

        Args:
            prompt: The full extraction prompt.

        Returns:
            Parsed dict from the model's JSON response.

        Raises:
            GeminiExtractionError: If all retries are exhausted or the response
                                   cannot be parsed as JSON.
        """
        client = self._get_client()
        last_exc: Exception = RuntimeError("No attempts made")
        delay = _RETRY_BASE_DELAY

        for attempt in range(_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self._temperature,
                        response_mime_type="application/json",
                    ),
                )
                return json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise GeminiExtractionError(
                    f"Gemini returned non-JSON response: {exc}"
                ) from exc
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Gemini call attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2

        raise GeminiExtractionError(
            f"All {_MAX_RETRIES} Gemini attempts failed: {last_exc}"
        ) from last_exc

    def _validate_and_coerce(self, raw: dict, chunk: Chunk) -> ChunkMetadata:
        """Validate the Gemini response dict against the schema and coerce types.

        Coercion is driven entirely by ``config/metadata_schema.json`` via
        ``SchemaVocabulary.coerce``:
          - String "2019" -> int 2019 for year_published / sample_size
          - Single string -> list[str] for array fields (keywords, modality, ...)
          - Enum values not in the schema -> the field's schema default
            (e.g. unknown domain -> "other", unknown doc_type -> "")
          - Keys not in the schema (legacy ``entities`` / ``evidence_level``) are
            dropped rather than passed through.

        Args:
            raw: Raw dict parsed from Gemini's JSON response.
            chunk: Source chunk (used to fill doc_id, chunk_index, page_start, etc.).

        Returns:
            Valid ChunkMetadata instance.

        Raises:
            ValidationError: If the response is structurally invalid after coercion.
                             Callers should catch this and call _fallback_extraction.
        """
        coerced = self._vocab.coerce(raw)

        # Always override provenance fields from chunk (never trust Gemini for these).
        coerced["doc_id"] = chunk.doc_id
        coerced["page_start"] = chunk.page_start
        coerced["page_end"] = chunk.page_end
        coerced["chunk_index"] = chunk.chunk_index

        return ChunkMetadata(**coerced)

    def _fallback_extraction(self, chunk: Chunk) -> ChunkMetadata:
        """Produce minimal metadata via rule-based heuristics when Gemini fails.

        Extracts:
          - doc_id, page_start, page_end, chunk_index from chunk fields
          - title from the first non-empty line of chunk text
          - keywords from high-frequency capitalized terms (TF-IDF not available here)
          - domain/doc_type left at their schema defaults ("other" / "") so the
            chunk is not misclassified; all remaining fields use ChunkMetadata's
            schema-aligned defaults. The removed biomedical fields (entities,
            evidence_level) are no longer populated.

        Args:
            chunk: The chunk for which Gemini extraction failed.

        Returns:
            ChunkMetadata with best-effort populated fields.
        """
        lines = [l.strip() for l in chunk.text.splitlines() if l.strip()]
        title = lines[0][:120] if lines else ""

        words = chunk.text.split()
        seen: set[str] = set()
        keywords: list[str] = []
        for word in words:
            cleaned = word.strip(".,;:!?()")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 3 and cleaned not in seen:
                seen.add(cleaned)
                keywords.append(cleaned)
                if len(keywords) >= 10:
                    break

        return ChunkMetadata(
            doc_id=chunk.doc_id,
            source_file="",
            title=title,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            keywords=keywords,
        )

    def _load_schema(self, schema_path: Optional[Path]) -> dict:
        """Load and return the metadata JSON schema.

        Args:
            schema_path: Explicit path (in production, ``settings.metadata_schema_path``
                         → ``config/rta_v1.json``). If None, resolves to the active RTA
                         schema at <project_root>/config/rta_v1.json.

        Returns:
            Parsed schema dict.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "config" / "rta_v1.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Metadata schema not found: {schema_path}")
        with open(schema_path) as f:
            return json.load(f)

    def _get_client(self) -> genai.Client:
        """Lazy-initialize and return the Vertex AI Gemini client.

        Uses Vertex AI (``vertexai=True``) with Application Default Credentials —
        the same auth as the Discovery Engine clients, no API key. The Vertex
        location is the raw ``gcp_location`` compute region (unlike Discovery
        Engine, which needs the derived multi-region).

        Returns:
            Configured ``google.genai.Client``.
        """
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_location,
            )
        return self._client


class GeminiExtractionError(Exception):
    """Raised when Gemini metadata extraction fails after all retries."""
