#!/usr/bin/env python
"""
Latency comparison: filtered-RAG (retrieved tagged guidance) vs. whole-corpus
RAG, against the same prompt and the same Gemini call parameters.

Two arms, run back to back against the Vertex AI Search DataStore populated
by scripts/batch_ingest.py:

  1. "filtered"    — Discovery Engine search, ranked, restricted to chunks
                      matching one or more --tag metadata filters (or
                      unfiltered-but-ranked if no --tag is given).
  2. "full_corpus" — every chunk in the DataStore, unranked, unfiltered —
                      the "use the entire corpus" condition.

Both arms build a prompt of the same shape (same instructions, same
question; only the retrieved context differs) and call the same Gemini
model with the same sampling/seed/thinking parameters, so the only
deliberate variable is retrieval strategy / context size. Retrieval time and
generation time are measured and reported separately, plus the total.

See rag_check/README.md for full usage and example invocations.

Usage:
    PYTHONPATH=. python rag_check/compare_latency.py \\
        --prompt "What should I do about a client disclosing suicidal ideation mid-session?" \\
        --tag session_event_tags=disclosure_SI \\
        --tag directionality=contraindicated \\
        --model gemini-2.5-flash

    # Entire-corpus-only sanity check (skip the filtered arm):
    PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --skip-filtered

    # No --tag at all: filtered arm ranks across the whole corpus (still
    # distinct from full_corpus, which sends every chunk unranked):
    PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings  # noqa: E402
from rag_check.llm_query import GenAIParams, run_query  # noqa: E402
from rag_check.retrieval import (  # noqa: E402
    FilterFieldError,
    RetrievedChunk,
    build_filter_expression,
    list_all_chunks,
    search_filtered,
)
from scripts._gcp_logging import log_api_error, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_SYSTEM_PREAMBLE = (
    "You are a clinical guidance assistant. Answer the question using ONLY "
    "the context passages below. If the context does not contain a relevant "
    "answer, say so explicitly rather than guessing.\n"
)


def parse_tag_args(tag_args: list[str]) -> list[tuple[str, str]]:
    """Parse repeated --tag field=value args into (field, value) pairs."""
    pairs = []
    for raw in tag_args:
        if "=" not in raw:
            raise SystemExit(f"Invalid --tag '{raw}', expected field=value")
        field_name, _, value = raw.partition("=")
        pairs.append((field_name.strip(), value.strip()))
    return pairs


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the final prompt: fixed instructions + numbered context + question.

    Identical shape for both arms so the comparison isolates retrieval/context
    size, not prompt wording.
    """
    if not chunks:
        context_block = "(no context retrieved)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            struct = chunk.struct_data
            source = struct.get("source_file", "unknown")
            page_s, page_e = struct.get("page_start"), struct.get("page_end")
            page = f"p.{page_s}" if page_s == page_e else f"pp.{page_s}-{page_e}"
            parts.append(f"[{i}] ({source}, {page})\n{chunk.text}")
        context_block = "\n\n".join(parts)

    return (
        f"{_SYSTEM_PREAMBLE}\n"
        f"--- CONTEXT ({len(chunks)} passage(s)) ---\n"
        f"{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {question}\n"
    )


def run_arm(
    name: str,
    chunks_fn,
    question: str,
    genai_params: GenAIParams,
) -> dict:
    """Time retrieval + generation for one arm and return a result record."""
    import time

    logger.info("--- Running arm: %s ---", name)
    retrieval_start = time.perf_counter()
    chunks = chunks_fn()
    retrieval_latency = time.perf_counter() - retrieval_start
    logger.info("[%s] retrieved %d chunk(s) in %.3fs", name, len(chunks), retrieval_latency)

    prompt = build_prompt(question, chunks)
    prompt_chars = len(prompt)

    result = run_query(prompt, genai_params)
    logger.info("[%s] generation latency: %.3fs", name, result.latency_seconds)

    return {
        "arm": name,
        "num_chunks": len(chunks),
        "chunk_ids": [c.chunk_id for c in chunks],
        "prompt_chars": prompt_chars,
        "retrieval_latency_seconds": round(retrieval_latency, 4),
        "generation_latency_seconds": round(result.latency_seconds, 4),
        "total_latency_seconds": round(retrieval_latency + result.latency_seconds, 4),
        "usage": result.usage,
        "answer_text": result.text,
    }


def run_comparison(
    prompt: str,
    *,
    search_query: Optional[str] = None,
    tag_pairs: Optional[list[tuple[str, str]]] = None,
    top_k: int = 10,
    max_corpus_chunks: Optional[int] = None,
    skip_filtered: bool = False,
    skip_full_corpus: bool = False,
    genai_params: GenAIParams,
) -> list[dict]:
    """Run one or both latency-comparison arms and return their result records.

    This is the reusable core behind both the single-run CLI (main(), below)
    and rag_check/param_sweep.py's per-row sweep — both just assemble inputs
    and hand them here, so a sweep run and a one-off CLI run can never drift
    apart in what they actually measure.

    Args:
        prompt: The question sent to the LLM (both arms).
        search_query: Search text for the filtered arm's ranking; defaults
            to `prompt`.
        tag_pairs: (field, value) pairs for the filtered arm's metadata
            filter — see retrieval.build_filter_expression(). Empty/None
            means an unfiltered-but-ranked search.
        top_k: Max chunks the filtered arm retrieves.
        max_corpus_chunks: Cap on chunks the full_corpus arm retrieves
            (None = no cap).
        skip_filtered / skip_full_corpus: Run only one arm. Not both True.
        genai_params: Model/sampling/thinking configuration, shared by
            whichever arm(s) run.

    Returns:
        A list with one result dict per arm actually run (see run_arm()).

    Raises:
        ValueError: If both skip flags are True.
        retrieval.FilterFieldError: If a tag_pairs field name isn't a known
            filterable field.
        Whatever the underlying Discovery Engine / genai calls raise.
    """
    if skip_filtered and skip_full_corpus:
        raise ValueError("skip_filtered and skip_full_corpus cannot both be set "
                          "(nothing left to run).")

    tag_pairs = tag_pairs or []
    filter_expr = build_filter_expression(tag_pairs)
    resolved_query = search_query or prompt

    records: list[dict] = []
    if not skip_filtered:
        records.append(run_arm(
            "filtered",
            lambda: search_filtered(resolved_query, filter_expr, top_k),
            prompt,
            genai_params,
        ))
    if not skip_full_corpus:
        records.append(run_arm(
            "full_corpus",
            lambda: list_all_chunks(max_corpus_chunks),
            prompt,
            genai_params,
        ))
    return records


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'arm':<14}{'chunks':>8}{'prompt_chars':>14}{'retrieval_s':>13}"
          f"{'generate_s':>12}{'total_s':>10}")
    print("-" * 78)
    for r in records:
        print(
            f"{r['arm']:<14}{r['num_chunks']:>8}{r['prompt_chars']:>14}"
            f"{r['retrieval_latency_seconds']:>13.3f}"
            f"{r['generation_latency_seconds']:>12.3f}"
            f"{r['total_latency_seconds']:>10.3f}"
        )
    print("=" * 78)
    if len(records) == 2:
        a, b = records[0], records[1]
        delta = b["total_latency_seconds"] - a["total_latency_seconds"]
        faster = a["arm"] if delta > 0 else b["arm"]
        print(f"\n'{faster}' was faster by {abs(delta):.3f}s total "
              f"({a['arm']}={a['total_latency_seconds']:.3f}s, "
              f"{b['arm']}={b['total_latency_seconds']:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LLM query latency: filtered-tag RAG vs. whole-corpus RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", required=True,
                        help="The question to ask the LLM (also used as the search query "
                             "for the filtered arm unless --search-query is given).")
    parser.add_argument("--search-query", default=None,
                        help="Search text for the filtered arm's ranking, if different "
                             "from --prompt (defaults to --prompt).")
    parser.add_argument("--tag", action="append", default=[],
                        metavar="FIELD=VALUE",
                        help="Metadata filter for the filtered arm, repeatable "
                             "(AND-combined), e.g. --tag session_event_tags=disclosure_SI. "
                             "Omit entirely to rank across the whole corpus unfiltered.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Max chunks retrieved by the filtered arm (default 10).")
    parser.add_argument("--max-corpus-chunks", type=int, default=None,
                        help="Cap on chunks retrieved by the full_corpus arm "
                             "(default: no cap, i.e. the literal entire corpus).")
    parser.add_argument("--skip-filtered", action="store_true",
                        help="Only run the full_corpus arm.")
    parser.add_argument("--skip-full-corpus", action="store_true",
                        help="Only run the filtered arm.")

    # genai inputs
    parser.add_argument("--model", default=None,
                        help="Gemini model id (default: settings.gemini_model_generation "
                             "/ GEMINI_MODEL_GENERATION from .env).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None, dest="sampling_top_p")
    parser.add_argument("--sampling-top-k", type=int, default=None,
                        help="Sampling top_k (distinct from --top-k, the retrieval count).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Thinking token budget (models that support it, e.g. 2.5 family). "
                             "0 disables thinking where supported; omit for model default.")

    parser.add_argument("--out", type=Path, default=None,
                        help="Write full JSON results (including answer text) to this path.")
    parser.add_argument("--verbose", action="store_true", help="Verbose/DEBUG logging.")
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.verbose)

    try:
        settings.validate_all()
    except Exception as exc:
        logger.error("Configuration problem:\n%s", exc)
        sys.exit(2)

    if args.skip_filtered and args.skip_full_corpus:
        raise SystemExit("--skip-filtered and --skip-full-corpus cannot both be set "
                          "(nothing left to run).")

    tag_pairs = parse_tag_args(args.tag)
    try:
        filter_expr = build_filter_expression(tag_pairs)
    except FilterFieldError as exc:
        logger.error(str(exc))
        sys.exit(2)

    search_query = args.search_query or args.prompt
    model = args.model or settings.gemini_model_generation
    genai_params = GenAIParams(
        model=model,
        temperature=args.temperature,
        top_p=args.sampling_top_p,
        top_k=args.sampling_top_k,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
    )
    logger.info(
        "Model=%s temperature=%s top_p=%s top_k=%s seed=%s max_output_tokens=%s "
        "thinking_budget=%s",
        model, args.temperature, args.sampling_top_p, args.sampling_top_k,
        args.seed, args.max_output_tokens, args.thinking_budget,
    )

    try:
        records = run_comparison(
            args.prompt,
            search_query=search_query,
            tag_pairs=tag_pairs,
            top_k=args.top_k,
            max_corpus_chunks=args.max_corpus_chunks,
            skip_filtered=args.skip_filtered,
            skip_full_corpus=args.skip_full_corpus,
            genai_params=genai_params,
        )
    except Exception as exc:  # noqa: BLE001 — surface any API/SDK error clearly
        log_api_error(logger, exc, "running the latency comparison")
        sys.exit(1)

    print_summary(records)

    if args.out:
        payload = {
            "prompt": args.prompt,
            "search_query": search_query,
            "tag_filters": tag_pairs,
            "filter_expression": filter_expr,
            "genai_params": vars(genai_params),
            "results": records,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
