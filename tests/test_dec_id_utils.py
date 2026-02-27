"""
Unit tests for shared DEC identifier helper utilities.

These tests lock parser/sorting behavior in `tests/dec_id_utils.py`, so higher-
level decision-log/compliance tests can rely on stable DEC-ID semantics.
"""

from __future__ import annotations

import unittest

try:
    from .dec_id_utils import DEC_ENTRY_RE, dec_sort_key, invalid_dec_ids, try_parse_dec_id
except ImportError:  # pragma: no cover
    from dec_id_utils import DEC_ENTRY_RE, dec_sort_key, invalid_dec_ids, try_parse_dec_id


class TestDecIdUtils(unittest.TestCase):
    def test_try_parse_dec_id_valid_and_invalid(self) -> None:
        self.assertEqual(try_parse_dec_id("DEC-001"), (1, ""))
        self.assertEqual(try_parse_dec_id("DEC-001a"), (1, "a"))
        self.assertEqual(try_parse_dec_id("DEC-010b"), (10, "b"))

        self.assertIsNone(try_parse_dec_id("DEC-1"))
        self.assertIsNone(try_parse_dec_id("DEC-001aa"))
        self.assertIsNone(try_parse_dec_id("FOO-001"))

    def test_invalid_dec_ids_filters_only_invalid_identifiers(self) -> None:
        ids = ["DEC-001", "DEC-001a", "DEC-1", "FOO-001", "DEC-010b", "DEC-001aa"]
        self.assertEqual(invalid_dec_ids(ids), ["DEC-1", "FOO-001", "DEC-001aa"])

    def test_dec_sort_key_preserves_canonical_order(self) -> None:
        dec_ids = ["DEC-010b", "DEC-001a", "DEC-002", "DEC-001"]
        self.assertEqual(
            sorted(dec_ids, key=dec_sort_key),
            ["DEC-001", "DEC-001a", "DEC-002", "DEC-010b"],
        )

    def test_dec_sort_key_raises_on_invalid_identifiers(self) -> None:
        for dec_id in ("DEC-1", "FOO-001", "DEC-001aa"):
            with self.assertRaises(ValueError):
                dec_sort_key(dec_id)

    def test_dec_entry_regex_extracts_dec_headers(self) -> None:
        text = """
DEC-001 — First decision
- Decision statement: ...

DEC-001a — First amendment
- Decision statement: ...

DEC-002 — Second decision
- Decision statement: ...
"""
        self.assertEqual(
            [m.group(1) for m in DEC_ENTRY_RE.finditer(text)],
            ["DEC-001", "DEC-001a", "DEC-002"],
        )


if __name__ == "__main__":
    unittest.main()
