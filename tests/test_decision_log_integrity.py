"""
Regression tests for decision-log structural integrity.

These tests enforce machine-checkable invariants on `research_pack/09_decision_log.md`
in the private source repository. In the public mirror, private research_pack
artifacts are intentionally absent and content checks are skipped.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PACK = ROOT / "research_pack"
DECISION_LOG_PATH = RESEARCH_PACK / "09_decision_log.md"

DEC_ID_RE = re.compile(r"^(DEC-(\d{3})([a-z]?))\s+—", flags=re.M)

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _dec_sort_key(dec_id: str) -> tuple[int, int]:
    match = re.match(r"DEC-(\d{3})([a-z]?)$", dec_id)
    if not match:
        raise ValueError(f"unexpected DEC id format: {dec_id}")
    number = int(match.group(1))
    suffix = match.group(2)
    suffix_rank = 0 if suffix == "" else (ord(suffix) - ord("a") + 1)
    return (number, suffix_rank)


class TestDecisionLogIntegrity(unittest.TestCase):
    def test_decision_log_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            return
        self.assertTrue(
            DECISION_LOG_PATH.exists(),
            msg=f"missing decision log in source repo: {DECISION_LOG_PATH}",
        )

    @unittest.skipUnless(
        DECISION_LOG_PATH.exists(),
        "decision log is absent in this repository scope",
    )
    def test_decision_ids_are_unique_and_base_ids_are_monotonic(self) -> None:
        text = DECISION_LOG_PATH.read_text(encoding="utf-8")
        ids = [m.group(1) for m in DEC_ID_RE.finditer(text)]
        self.assertGreater(len(ids), 0, msg="decision log must contain DEC entries")

        self.assertEqual(
            len(ids),
            len(set(ids)),
            msg="decision log contains duplicate DEC identifiers",
        )

        base_ids = [dec_id for dec_id in ids if re.match(r"DEC-\d{3}$", dec_id)]
        ordered = sorted(base_ids, key=_dec_sort_key)
        self.assertEqual(
            base_ids,
            ordered,
            msg=(
                "decision log base DEC identifiers must be in monotonic "
                "ascending order"
            ),
        )

    @unittest.skipUnless(
        DECISION_LOG_PATH.exists(),
        "decision log is absent in this repository scope",
    )
    def test_decision_amendments_have_base_entries(self) -> None:
        text = DECISION_LOG_PATH.read_text(encoding="utf-8")
        entries = [m.group(1) for m in DEC_ID_RE.finditer(text)]
        ids = set(entries)
        position = {dec_id: idx for idx, dec_id in enumerate(entries)}
        amendments = sorted(
            dec_id for dec_id in ids if re.match(r"DEC-\d{3}[a-z]$", dec_id)
        )

        for amendment in amendments:
            base = amendment[:-1]
            self.assertIn(
                base,
                ids,
                msg=f"amendment {amendment} has no base decision entry {base}",
            )
            self.assertGreater(
                position[amendment],
                position[base],
                msg=f"amendment {amendment} must appear after base decision {base}",
            )


if __name__ == "__main__":
    unittest.main()
