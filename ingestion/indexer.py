"""
Vertex AI Search (Discovery Engine) DataStore indexer.

Imports chunk JSONL from GCS into the configured DataStore via the
ImportDocuments API. Import operations are asynchronous LROs (long-running
operations) — use wait_for_import() to poll until completion.

Important notes:
  - Schema must be registered (scripts/setup_vertex_search.py) BEFORE importing.
  - ImportDocuments is not atomic: partial success is possible. Always check
    ImportResult.failure_count after waiting.
  - Re-importing the same doc_id will create duplicate entries. Use
    delete_document() before re-importing updated documents.

See caveats.md §1 for full Vertex AI Search limitations.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from google.api_core import retry as api_retry
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable
from google.cloud import discoveryengine_v1 as discoveryengine
from google.longrunning.operations_pb2 import GetOperationRequest

from models import ImportResult

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 15
_DEFAULT_TIMEOUT_SECONDS = 600
_RPC_TIMEOUT = 300
_RETRY_TRANSIENT = api_retry.Retry(
    predicate=api_retry.if_exception_type(DeadlineExceeded, ServiceUnavailable),
    initial=2.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=120.0,
)


class VertexSearchIndexer:
    """
    Manages document import and deletion in a Vertex AI Search DataStore.

    Usage:
        indexer = VertexSearchIndexer(
            project_id="my-project",
            location="global",
            datastore_id="my-rag-datastore",
        )
        op_name = indexer.import_chunks("gs://bucket/chunks/abc123.jsonl", "abc123")
        result = indexer.wait_for_import(op_name)
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        datastore_id: str,
        api_endpoint: Optional[str] = None,
    ) -> None:
        """
        Args:
            project_id: GCP project ID.
            location: DataStore location — one of "global"/"us"/"eu", matching the
                      region chosen at DataStore creation (immutable). Pass
                      ``settings.discovery_engine_location``, NOT the raw
                      ``gcp_location`` (a compute region like "us-central1" is not
                      a valid Discovery Engine location and will be misrouted).
            datastore_id: The DataStore resource ID (not the full resource name).
            api_endpoint: Regional endpoint for non-global locations, e.g.
                      "us-discoveryengine.googleapis.com"
                      (``settings.discovery_engine_endpoint``). None uses the
                      default global endpoint. A wrong/absent endpoint causes the
                      "endpoint can only serve global region" INVALID_ARGUMENT.
        """
        self._project_id = project_id
        self._location = location
        self._datastore_id = datastore_id
        self._api_endpoint = api_endpoint
        self._client: Optional[discoveryengine.DocumentServiceClient] = None

    @property
    def _datastore_name(self) -> str:
        """Return the fully-qualified DataStore resource name."""
        return (
            f"projects/{self._project_id}/locations/{self._location}"
            f"/collections/default_collection/dataStores/{self._datastore_id}"
        )

    def import_chunks(self, gcs_jsonl_uri: str, doc_id: str) -> str:
        """Submit a GCS-sourced import request and return the LRO operation name.

        Constructs a GcsSource pointing to the JSONL file and calls
        ImportDocuments with FULL reconciliation mode (safer for re-imports
        than INCREMENTAL — avoids partial state on retry).

        Args:
            gcs_jsonl_uri: Full gs:// URI to the chunk JSONL file,
                           e.g. "gs://bucket/chunks/abc123.jsonl".
            doc_id: The document ID (used for logging and error tracking only;
                    the DataStore document IDs come from the JSONL content).

        Returns:
            LRO operation name string (used with wait_for_import).

        Raises:
            google.api_core.exceptions.GoogleAPIError: On request failure.
        """
        client = self._get_client()
        parent = f"{self._datastore_name}/branches/default_branch"

        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            gcs_source=discoveryengine.GcsSource(
                input_uris=[gcs_jsonl_uri],
                data_schema="document",
            ),
            reconciliation_mode=(
                discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL
            ),
        )

        operation = client.import_documents(
            request=request,
            timeout=_RPC_TIMEOUT,
            retry=_RETRY_TRANSIENT,
        )
        op_name = operation.operation.name
        logger.info("Import LRO started for doc %s: %s", doc_id, op_name)
        return op_name

    def wait_for_import(
        self,
        operation_name: str,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> ImportResult:
        """Poll an import LRO until completion or timeout.

        Logs progress at each poll interval. Returns an ImportResult with
        success/failure counts from the operation metadata.

        Args:
            operation_name: LRO name returned by import_chunks().
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            ImportResult with completed=True and populated counts.

        Raises:
            TimeoutError: If the operation does not complete within `timeout`.
            ImportError: If the operation completes with a terminal error status.
        """
        client = self._get_client()
        ops_client = client._transport.operations_client
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Import LRO {operation_name} timed out after {timeout}s"
                )

            op = ops_client.get_operation(
                GetOperationRequest(name=operation_name),
                timeout=_RPC_TIMEOUT,
            )

            if op.done:
                if op.HasField("error"):
                    raise ImportError(
                        f"Import LRO failed: {op.error.message} (code {op.error.code})"
                    )
                metadata = discoveryengine.ImportDocumentsMetadata()
                op.metadata.Unpack(metadata)
                return self._parse_import_result(
                    {
                        "successCount": metadata.success_count,
                        "failureCount": metadata.failure_count,
                        "errorSamples": list(metadata.error_samples),
                    }
                )

            logger.info(
                "Import in progress for %s (%.0fs elapsed)...", operation_name, elapsed
            )
            time.sleep(poll_interval)

    def delete_document(self, chunk_id: str) -> None:
        """Delete a single document (chunk) from the DataStore by its ID.

        Use this before re-importing updated chunks to avoid duplicates.
        For bulk re-ingestion, use scripts/purge_datastore.py instead.

        Args:
            chunk_id: The DataStore document ID (same as Chunk.chunk_id).

        Raises:
            google.api_core.exceptions.NotFound: If the document does not exist.
        """
        client = self._get_client()
        name = (
            f"{self._datastore_name}/branches/default_branch/documents/{chunk_id}"
        )
        client.delete_document(
            name=name,
            timeout=_RPC_TIMEOUT,
            retry=_RETRY_TRANSIENT,
        )
        logger.debug("Deleted document: %s", chunk_id)

    def list_documents(self, page_size: int = 100) -> list[str]:
        """List all document IDs currently in the DataStore.

        Handles pagination automatically. Useful for auditing or deduplication checks.

        Args:
            page_size: Number of documents per API page (max 1000).

        Returns:
            List of document ID strings.
        """
        client = self._get_client()
        parent = f"{self._datastore_name}/branches/default_branch"
        request = discoveryengine.ListDocumentsRequest(
            parent=parent, page_size=page_size
        )
        return [
            doc.id
            for doc in client.list_documents(
                request=request,
                timeout=_RPC_TIMEOUT,
                retry=_RETRY_TRANSIENT,
            )
        ]

    def _get_client(self) -> discoveryengine.DocumentServiceClient:
        """Lazy-initialize and return the Discovery Engine document service client.

        Targets the regional endpoint when ``api_endpoint`` was provided, so
        requests for a non-global DataStore are not sent to the global endpoint.

        Returns:
            Authenticated DocumentServiceClient.
        """
        if self._client is None:
            client_options = (
                ClientOptions(api_endpoint=self._api_endpoint)
                if self._api_endpoint
                else None
            )
            self._client = discoveryengine.DocumentServiceClient(
                client_options=client_options
            )
        return self._client

    def _parse_import_result(self, operation_metadata: dict) -> ImportResult:
        """Extract success/failure counts from a completed import operation's metadata.

        Args:
            operation_metadata: The metadata dict from the completed LRO.

        Returns:
            ImportResult with counts populated from metadata fields.
        """
        return ImportResult(
            operation_name="",
            success_count=int(operation_metadata.get("successCount") or 0),
            failure_count=int(operation_metadata.get("failureCount") or 0),
            errors=[str(e) for e in operation_metadata.get("errorSamples", [])],
            completed=True,
        )
