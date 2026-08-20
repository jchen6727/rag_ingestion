"""
Pluggable chunk-processing strategies for the ingestion metadata stage.

A strategy decides two things, and nothing else:

  1. **Units** — how chunks are grouped into "units". Chunks *within* a unit are
     processed **sequentially** (so a later chunk may depend on earlier ones);
     units are processed **concurrently**, up to ``max_concurrency``.
  2. **Context** — what ``context_window`` string each chunk is tagged with
     (e.g. its chapter heading + the preceding chunks in the same chapter).

This is the extension point requested in `devlog.md#processing-strategem`. To add
a new behaviour (e.g. "summarize the chapter first, then tag each chunk with that
summary"), implement a `ProcessingStrategy` subclass and register it in
`_STRATEGIES` — the ingest loop (`scripts/batch_ingest.py`) does not change.

Chosen defaults (per project decision 2026-07-27):
  - default strategy = ``chapter`` (tags see chapter context),
  - concurrency across chapters, sequential within a chapter.

Set via env / .env: ``INGEST_STRATEGY`` and ``INGEST_CONCURRENCY``
(read through `config.settings`).
"""

from __future__ import annotations

from models import Chunk


class ProcessingStrategy:
    """Base class. A strategy is stateless; it only partitions and describes context."""

    name: str = "base"
    max_concurrency: int = 1

    def units(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Partition chunks into units (processed sequentially within, concurrently across)."""
        raise NotImplementedError

    def context_for(self, chunk: Chunk, unit: list[Chunk], index_in_unit: int) -> str:
        """Return the context_window string for ``chunk`` (its position in its unit)."""
        return ""


class IndependentStrategy(ProcessingStrategy):
    """Every chunk is its own unit → maximum concurrency, no cross-chunk context.

    This is the fastest option and is correct when tagging does not depend on
    neighbouring chunks.
    """

    name = "independent"

    def __init__(self, max_concurrency: int = 8) -> None:
        self.max_concurrency = max(1, max_concurrency)

    def units(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [[c] for c in chunks]

    def context_for(self, chunk: Chunk, unit: list[Chunk], index_in_unit: int) -> str:
        return ""


class ChapterContextStrategy(ProcessingStrategy):
    """Group consecutive chunks by chapter (``parent_section``); tag with chapter context.

    - Units = runs of consecutive chunks sharing a ``parent_section``. Different
      chapters run concurrently (up to ``max_concurrency``); chunks inside one
      chapter run in order, so each chunk's context can include the ones before it.
    - Context = the chapter heading plus the most recent preceding text in the
      chapter, truncated to ``context_char_budget`` characters.
    """

    name = "chapter"

    def __init__(self, max_concurrency: int = 4, context_char_budget: int = 1500) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.context_char_budget = context_char_budget

    def units(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        units: list[list[Chunk]] = []
        current: list[Chunk] = []
        current_key: str | None = None
        for c in chunks:
            key = c.parent_section or ""
            if current and key != current_key:
                units.append(current)
                current = []
            current.append(c)
            current_key = key
        if current:
            units.append(current)
        return units

    def context_for(self, chunk: Chunk, unit: list[Chunk], index_in_unit: int) -> str:
        parts: list[str] = []
        heading = chunk.parent_section or ""
        if heading:
            parts.append(f"Chapter/section: {heading}")
        preceding = " ".join(p.text for p in unit[:index_in_unit])
        if preceding:
            # Keep the most recent context if it overflows the budget.
            if len(preceding) > self.context_char_budget:
                preceding = preceding[-self.context_char_budget:]
            parts.append(preceding)
        return "\n".join(parts)


# Registry — add new strategies here.
_STRATEGIES: dict[str, type[ProcessingStrategy]] = {
    "independent": IndependentStrategy,
    "chapter": ChapterContextStrategy,
}


def build_strategy(name: str, max_concurrency: int) -> ProcessingStrategy:
    """Construct a strategy by name.

    Args:
        name: One of ``_STRATEGIES`` ("independent" | "chapter").
        max_concurrency: Upper bound on concurrent units.

    Returns:
        A ready-to-use ProcessingStrategy.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    cls = _STRATEGIES.get((name or "").lower())
    if cls is None:
        raise ValueError(
            f"Unknown ingest strategy '{name}'. Known: {sorted(_STRATEGIES)}"
        )
    return cls(max_concurrency=max_concurrency)
