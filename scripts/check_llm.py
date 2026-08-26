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

    # List models actually servable in this Vertex location (one real ping per
    # candidate model — not just what the model catalog lists):
    PYTHONPATH=. python scripts/check_llm.py --list
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            # client.models.list() queries a global model-garden catalog, NOT
            # per-location deployment — it lists a model as present even when
            # it isn't actually servable in settings.gcp_location (confirmed
            # 2026-08-20: gemini-3.6-flash showed up here for 'us-central1' but
            # 404'd on a real call). So this makes one minimal generate_content
            # ping per candidate model — the only reliable availability signal
            # — concurrently, and only prints the ones that actually work.
            candidates = []
            for m in client.models.list():
                name = m.name.rsplit("/", 1)[-1]
                # The catalog also lists non-chat models (embeddings, Veo, Imagen,
                # AutoML, medical/vision foundation models, ...) that don't take
                # generateContent at all and would just waste a probe call and
                # clutter the output. metadata_gen only ever uses a "gemini-*"
                # text model, so restrict probing to those.
                if not name.startswith("gemini-"):
                    continue
                if "tts" in name or "audio" in name or "image" in name or "embedding" in name:
                    continue
                candidates.append(name)

            if not candidates:
                print(f"No candidate models returned for '{settings.gcp_project_id}' — "
                      f"check the location and that Vertex/Gemini is enabled.")
                return

            print(f"Checking {len(candidates)} candidate model(s) against Vertex location "
                  f"'{settings.gcp_location}' (this makes one real call per model)...")

            def _probe(name: str) -> tuple[str, bool, str]:
                try:
                    resp = client.models.generate_content(
                        model=name,
                        contents="ping",
                        config=types.GenerateContentConfig(max_output_tokens=1, temperature=0.0),
                    )
                    _ = resp.text
                    return name, True, ""
                except Exception as exc:  # noqa: BLE001
                    return name, False, str(exc)

            results = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_probe, name) for name in candidates]
                for fut in as_completed(futures):
                    results.append(fut.result())
            results.sort(key=lambda r: r[0])

            servable = [r for r in results if r[1]]
            unservable = [r for r in results if not r[1]]

            print(f"\nServable in '{settings.gcp_location}' ({len(servable)}/{len(results)}):")
            for name, _ok, _err in servable:
                print(f"  ✓ {name}")
            if not servable:
                print("  (none — check the location and that Vertex/Gemini is enabled)")

            if unservable:
                print(f"\nListed but NOT servable here ({len(unservable)}) — "
                      f"present in the model catalog but 404s on a real call:")
                for name, _ok, err in unservable:
                    if args.verbose:
                        print(f"  ✗ {name}: {err}")
                    else:
                        print(f"  ✗ {name}")
                if not args.verbose:
                    print("  (re-run with --verbose for the underlying error on each)")
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
