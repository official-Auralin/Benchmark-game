"""
Shared DEC identifier helpers for regression tests.

Centralizing DEC patterns/helpers keeps ordering and parsing semantics
consistent across test modules that reason about decision-log entries.
"""

from __future__ import annotations

import re


DEC_ID_PATTERN = r"DEC-\d{3}[a-z]?"
DEC_ID_PARSE_RE = re.compile(r"^DEC-(\d{3})([a-z]?)$")
DEC_ID_FULL_RE = re.compile(rf"^{DEC_ID_PATTERN}$")
DEC_ENTRY_RE = re.compile(rf"^({DEC_ID_PATTERN})\s+—", flags=re.M)
DEC_REFERENCE_RE = re.compile(r"DEC-\d{3}[a-z]?")
BASE_DEC_RE = re.compile(r"DEC-\d{3}")


def try_parse_dec_id(dec_id: str) -> tuple[int, str] | None:
    """Return parsed (number, suffix) if valid, else None."""
    match = DEC_ID_PARSE_RE.fullmatch(dec_id)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def dec_sort_key(dec_id: str) -> tuple[int, int]:
    """
    Stable sort key for DEC IDs.

    Raises ValueError for invalid identifiers so callers can enforce strict
    input hygiene in tests.
    """
    parsed = try_parse_dec_id(dec_id)
    if parsed is None:
        raise ValueError(f"invalid DEC identifier: {dec_id}")
    number, suffix = parsed
    suffix_rank = 0 if suffix == "" else ord(suffix) - ord("a") + 1
    return (number, suffix_rank)


def invalid_dec_ids(ids: list[str]) -> list[str]:
    """Return DEC identifiers that do not match canonical DEC format."""
    return [dec_id for dec_id in ids if try_parse_dec_id(dec_id) is None]
