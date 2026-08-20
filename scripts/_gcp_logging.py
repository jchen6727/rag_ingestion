"""
Shared logging setup and Google API error diagnostics for the ingestion and
evaluation scripts.

Two things every script here wants:

  1. `setup_logging(verbose)` — one consistent log format, quiet-by-default noisy
     Google/gRPC loggers, and a `--verbose` switch that turns them all up to DEBUG
     when you are trying to see exactly what an API call sent and received.

  2. `describe_google_error(exc)` — turns an opaque Google exception
     (`PermissionDenied`, `NotFound`, `ResourceExhausted`, a bare
     `DefaultCredentialsError`, ...) into a short, actionable message that tells a
     non-expert operator what to actually *do* about it.

Import from the scripts:

    from scripts._gcp_logging import setup_logging, describe_google_error, log_api_error
"""

from __future__ import annotations

import logging
import sys
import warnings

# The bundled google.generativeai package prints a noisy end-of-life FutureWarning
# on import. It is not actionable by an operator, so we hide it to keep the output
# readable. (Tracked separately as a dependency-upgrade task.)
# `message` is matched with re.match, so we need (?s) DOTALL because the warning text
# starts with newlines; we also filter by the issuing module as a belt-and-braces.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.generativeai.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*generativeai.*")

# Loggers that are extremely chatty at DEBUG. Kept at WARNING unless --verbose.
_NOISY_LOGGERS = (
    "google",
    "google.auth",
    "google.api_core",
    "google.cloud",
    "grpc",
    "urllib3",
    "httpx",             # sentence-transformers/huggingface model download chatter
    "httpcore",
    "huggingface_hub",
    "filelock",
    "pdfminer",          # pdfplumber's backend — floods DEBUG with per-glyph logs
    "sentence_transformers",
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logging and return the shared 'ingestion' logger.

    Args:
        verbose: If True, everything (including Google/gRPC internals) logs at
            DEBUG — use this to debug API request/response details. If False,
            our own scripts log at INFO and the noisy libraries stay at WARNING.

    Returns:
        The ``ingestion`` logger, ready to use.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,  # override any prior basicConfig from an imported module
    )
    lib_level = logging.DEBUG if verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(lib_level)
    return logging.getLogger("ingestion")


# Maps a substring of the exception class name -> what to do about it.
# Ordered most-specific first; the first match wins.
_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (
        "DefaultCredentialsError",
        "No Google credentials were found. Log in with:\n"
        "      gcloud auth application-default login",
    ),
    (
        "Unauthenticated",
        "Your credentials are missing or expired. Refresh them with:\n"
        "      gcloud auth application-default login",
    ),
    (
        "PermissionDenied",
        "Your account is authenticated but lacks an IAM permission for this call.\n"
        "      Run  scripts/preflight_check.sh  — it lists the exact roles/commands to fix this.",
    ),
    (
        "NotFound",
        "The resource does not exist. If this is the DataStore or Engine, run\n"
        "      PYTHONPATH=. python scripts/setup_vertex_search.py\n"
        "      first, and double-check GCP_LOCATION and VERTEX_SEARCH_DATASTORE_ID in your .env.",
    ),
    (
        "AlreadyExists",
        "The resource already exists — this is usually safe to ignore (setup is idempotent).",
    ),
    (
        "ResourceExhausted",
        "You hit a quota or rate limit (common with Gemini or Discovery Engine).\n"
        "      Wait a minute and retry, process fewer chunks with --limit, or request a quota increase.",
    ),
    (
        "InvalidArgument",
        "The request was malformed or a value/region is wrong.\n"
        "      Check GCP_LOCATION and that the schema was registered by setup_vertex_search.py.",
    ),
    (
        "FailedPrecondition",
        "A precondition failed — most often a required API is not enabled or billing is off.\n"
        "      Run  scripts/preflight_check.sh  to check APIs and billing.",
    ),
    (
        "DeadlineExceeded",
        "The API call timed out (slow network or a long operation). Retry.",
    ),
    (
        "ServiceUnavailable",
        "The Google service is temporarily unavailable. Wait a moment and retry.",
    ),
)

# Substrings sometimes found in the message text rather than the class name.
_MESSAGE_HINTS: tuple[tuple[str, str], ...] = (
    ("has not been used", "The required Google API is not enabled. Enable it (see scripts/preflight_check.sh)."),
    ("SERVICE_DISABLED", "The required Google API is not enabled. Enable it (see scripts/preflight_check.sh)."),
    ("billing", "Billing may be disabled on the project. Enable billing, then retry."),
    ("credential", "Gemini/Vertex could not authenticate. Run  gcloud auth application-default login  and set GCP_PROJECT_ID in .env."),
)


def describe_google_error(exc: BaseException) -> str:
    """Return a short, actionable explanation for a Google API / auth exception.

    Falls back to the raw class name and message for unrecognized errors, so no
    information is ever lost — the hint is added on top when we recognize it.

    Args:
        exc: The caught exception.

    Returns:
        A multi-line human-readable string (safe to print or log).
    """
    name = type(exc).__name__
    msg = str(exc).strip()

    hint = None
    for needle, text in _ERROR_HINTS:
        if needle in name:
            hint = text
            break
    if hint is None:
        for needle, text in _MESSAGE_HINTS:
            if needle.lower() in msg.lower():
                hint = text
                break

    lines = [f"Google API error [{name}]: {msg or '(no message)'}"]
    if hint:
        lines.append(f"  → {hint}")
    else:
        lines.append("  → Unrecognized error. Re-run with --verbose for the full request/response.")
    return "\n".join(lines)


def log_api_error(logger: logging.Logger, exc: BaseException, context: str = "") -> None:
    """Log a Google API error with its actionable explanation.

    Args:
        logger: Logger to emit on.
        exc: The caught exception.
        context: Optional short description of what was being attempted
            (e.g. "importing chunks into the DataStore").
    """
    where = f" while {context}" if context else ""
    logger.error("Failed%s.\n%s", where, describe_google_error(exc))
    logger.debug("Full traceback:", exc_info=exc)
