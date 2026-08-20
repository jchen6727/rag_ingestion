"""
Check that the configured Gemini model is actually reachable and usable BEFORE
you kick off a long ingestion. Google is retiring older Gemini models, so a model
that worked last month may now return NOT_FOUND — this catches that in one quick
call instead of failing mid-ingest.

Auth: Vertex AI via Application Default Credentials (same as the rest of the
pipeline — no API key). Run `gcloud auth application-default login` and set
GCP_PROJECT_ID / GCP_LOCATION in .env first.

Usage:
    # Check the model metadata_gen will use (settings.gemini_model_metadata):
    PYTHONPATH=. python scripts/check_llm.py

    # Check a specific model:
    PYTHONPATH=. python scripts/check_llm.py --model gemini-1.5-pro

    # List models your project can use in this Vertex location:
    PYTHONPATH=. python scripts/check_llm.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._gcp_logging import describe_google_error, setup_logging

from config.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Gemini model is available on Vertex AI.")
    parser.add_argument("--model", default=None,
                        help="Model id to check (default: settings.gemini_model_metadata).")
    parser.add_argument("--list", action="store_true",
                        help="List models available to your project in this Vertex location.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    model = args.model or settings.gemini_model_metadata

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai is not installed. Run: pip install -r requirements.txt")
        sys.exit(2)

    try:
        settings.validate_all()
    except Exception as exc:  # missing GCP_PROJECT_ID etc.
        logger.error("Configuration problem — check your .env:\n%s", exc)
        sys.exit(2)

    try:
        client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not create the Vertex AI client.\n%s", describe_google_error(exc))
        sys.exit(1)

    try:
        if args.list:
            print(f"Models available to '{settings.gcp_project_id}' in Vertex location "
                  f"'{settings.gcp_location}':")
            found = False
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
                if (not actions) or ("generateContent" in actions):
                    print(f"  {m.name}")
                    found = True
            if not found:
                print("  (none returned — check the location and that Vertex/Gemini is enabled)")
            return

        # Definitive check: a minimal generation. Confirms the model is reachable
        # AND usable with the current ADC, in one call.
        resp = client.models.generate_content(
            model=model,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=1, temperature=0.0),
        )
        _ = resp.text  # touch the response to surface any decoding issue
        print(f"Model: {model}")
        print("  ✓ Reachable and usable for metadata generation (Vertex AI + ADC).")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not use model '%s'.\n%s", model, describe_google_error(exc))
        print("\nTip: run  PYTHONPATH=. python scripts/check_llm.py --list  to see available models,")
        print("then set GEMINI_MODEL_METADATA in .env to one of them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
