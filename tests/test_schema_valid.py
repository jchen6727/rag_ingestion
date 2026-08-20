"""
CI parse gate for JSON schema files (summary.md §2.2).

Any commit touching a config/*.json schema must fail CI if it does not parse.
This defect class (e.g. a stray trailing comma) should never reach human review.

Run:
    pytest tests/test_schema_valid.py -v
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_JSON_SCHEMAS = sorted(glob.glob(str(_ROOT / "config" / "*.json")))


@pytest.mark.parametrize("path", _JSON_SCHEMAS)
def test_schema_parses(path: str) -> None:
    """Every config/*.json must be valid JSON."""
    with open(path) as f:
        json.load(f)


def test_schema_files_present() -> None:
    """Guard against the glob silently matching nothing."""
    names = {Path(p).name for p in _JSON_SCHEMAS}
    assert "rta_v1.json" in names           # active RTA schema
    assert "metadata_schema.json" in names  # preserved ASA/historical schema
