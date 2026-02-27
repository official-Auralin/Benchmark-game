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
    from .dec_id_utils import BASE_DEC_RE, DEC_ENTRY_RE, DEC_REFERENCE_RE, dec_sort_key, invalid_dec_ids
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover
    from dec_id_utils import BASE_DEC_RE, DEC_ENTRY_RE, DEC_REFERENCE_RE, dec_sort_key, invalid_dec_ids
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PACK = ROOT / "research_pack"
AUDIT_PATH = RESEARCH_PACK / "55_phase_ah_full_prompt_compliance_audit.md"
DECISION_LOG_PATH = RESEARCH_PACK / "09_decision_log.md"

ANCHOR_DECISIONS_LINE_RE = re.compile(
    r"^- Decisions:\s+`DEC-001`\.\.`DEC-(\d{3})`\s+\(plus amendments (.+)\)",
    flags=re.M,
)

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


class TestDecSortKeyHelper(unittest.TestCase):
    def test_dec_sort_key_sort_order(self) -> None:
        dec_ids = ["DEC-010b", "DEC-001a", "DEC-002", "DEC-001"]
        self.assertEqual(
            sorted(dec_ids, key=dec_sort_key),
            ["DEC-001", "DEC-001a", "DEC-002", "DEC-010b"],
        )

    def test_dec_sort_key_invalid_ids_raise_value_error(self) -> None:
        invalid_ids = ["DEC-1", "FOO-001", "DEC-001aa"]
        for dec_id in invalid_ids:
            with self.assertRaises(ValueError):
                dec_sort_key(dec_id)


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
        amendments_in_audit = DEC_REFERENCE_RE.findall(line_match.group(2))
        self.assertGreater(
            len(amendments_in_audit),
            0,
            msg="compliance audit decisions line should list amendment identifiers",
        )
        malformed_amendments = invalid_dec_ids(amendments_in_audit)
        self.assertFalse(
            malformed_amendments,
            msg=f"compliance audit contains malformed DEC identifiers: {malformed_amendments}",
        )

        decision_ids = DEC_ENTRY_RE.findall(decision_log_text)
        self.assertGreater(len(decision_ids), 0, msg="decision log must contain DEC entries")
        malformed_decisions = invalid_dec_ids(decision_ids)
        self.assertFalse(
            malformed_decisions,
            msg=f"decision log contains malformed DEC identifiers: {malformed_decisions}",
        )

        base_ids = [dec_id for dec_id in decision_ids if BASE_DEC_RE.fullmatch(dec_id)]
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
        missing_amendments = sorted(set(amendments_in_audit) - decision_id_set, key=dec_sort_key)
        self.assertFalse(
            missing_amendments,
            msg=f"amendments listed in compliance audit missing from decision log: {missing_amendments}",
        )

        non_amendments_listed = sorted(
            [dec_id for dec_id in amendments_in_audit if BASE_DEC_RE.fullmatch(dec_id)],
            key=dec_sort_key,
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
            sorted(amendments_in_audit, key=dec_sort_key),
            msg="compliance audit amendment list should be sorted by DEC identifier order",
        )


if __name__ == "__main__":
    unittest.main()
