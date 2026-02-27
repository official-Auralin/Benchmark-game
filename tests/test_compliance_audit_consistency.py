"""
Regression tests for compliance-audit consistency.

In the private source repository, these checks ensure that summary metadata in
`research_pack/55_phase_ah_full_prompt_compliance_audit.md` stays consistent
with canonical sources such as `research_pack/09_decision_log.md`. In the
public mirror, private research-pack artifacts are intentionally absent and
content checks are skipped after scope validation.
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
AUDIT_PATH = RESEARCH_PACK / "55_phase_ah_full_prompt_compliance_audit.md"
DECISION_LOG_PATH = RESEARCH_PACK / "09_decision_log.md"

ANCHOR_DECISIONS_LINE_RE = re.compile(
    r"^- Decisions:\s+`DEC-001`\.\.`DEC-(\d{3})`\s+\(plus amendments (.+)\)",
    flags=re.M,
)
DEC_ID_RE = re.compile(r"DEC-\d{3}[a-z]?")
DEC_ENTRY_RE = re.compile(r"^(DEC-\d{3}[a-z]?)\s+—", flags=re.M)

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _dec_sort_key(dec_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"DEC-(\d{3})([a-z]?)", dec_id)
    if match is None:
        raise ValueError(f"invalid DEC identifier: {dec_id}")
    number = int(match.group(1))
    suffix = match.group(2)
    suffix_rank = 0 if suffix == "" else ord(suffix) - ord("a") + 1
    return (number, suffix_rank)


class TestComplianceAuditConsistency(unittest.TestCase):
    def test_research_pack_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            return

        self.assertTrue(AUDIT_PATH.exists(), msg=f"missing compliance audit: {AUDIT_PATH}")
        self.assertTrue(
            DECISION_LOG_PATH.exists(),
            msg=f"missing decision log: {DECISION_LOG_PATH}",
        )

    @unittest.skipUnless(
        AUDIT_PATH.exists() and DECISION_LOG_PATH.exists(),
        "required research-pack files are absent in this repository scope",
    )
    def test_anchor_decision_range_and_amendments_match_decision_log(self) -> None:
        audit_text = AUDIT_PATH.read_text(encoding="utf-8")
        decision_log_text = DECISION_LOG_PATH.read_text(encoding="utf-8")

        line_match = ANCHOR_DECISIONS_LINE_RE.search(audit_text)
        self.assertIsNotNone(
            line_match,
            msg=(
                "compliance audit anchors section must include a decisions line "
                "with DEC range and amendment list"
            ),
        )
        if line_match is None:
            return

        max_dec_in_audit = int(line_match.group(1))
        amendments_in_audit = DEC_ID_RE.findall(line_match.group(2))
        self.assertGreater(
            len(amendments_in_audit),
            0,
            msg="compliance audit decisions line should list amendment identifiers",
        )

        decision_ids = DEC_ENTRY_RE.findall(decision_log_text)
        self.assertGreater(len(decision_ids), 0, msg="decision log must contain DEC entries")
        base_ids = [dec_id for dec_id in decision_ids if re.fullmatch(r"DEC-\d{3}", dec_id)]
        self.assertGreater(len(base_ids), 0, msg="decision log must contain base DEC entries")
        max_base_dec = max(int(dec_id.split("-")[1]) for dec_id in base_ids)

        self.assertEqual(
            max_dec_in_audit,
            max_base_dec,
            msg=(
                "compliance audit decision range upper bound must match the highest "
                f"base DEC in decision log (expected DEC-{max_base_dec:03d})"
            ),
        )

        decision_id_set = set(decision_ids)
        missing_amendments = sorted(set(amendments_in_audit) - decision_id_set, key=_dec_sort_key)
        self.assertFalse(
            missing_amendments,
            msg=f"amendments listed in compliance audit missing from decision log: {missing_amendments}",
        )

        non_amendments_listed = sorted(
            [dec_id for dec_id in amendments_in_audit if re.fullmatch(r"DEC-\d{3}", dec_id)],
            key=_dec_sort_key,
        )
        self.assertFalse(
            non_amendments_listed,
            msg=(
                "compliance audit amendment list should include only amendment IDs "
                f"(found base IDs: {non_amendments_listed})"
            ),
        )

        self.assertEqual(
            amendments_in_audit,
            sorted(amendments_in_audit, key=_dec_sort_key),
            msg="compliance audit amendment list should be sorted by DEC identifier order",
        )


if __name__ == "__main__":
    unittest.main()
