"""
Regression tests for CI policy workflow invariants.

These tests ensure the repository workflow keeps required benchmark-governance
checks enabled in CI. They guard against accidental removal of the integrated
release-candidate enforcement path.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "License Name"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "gf01-gate.yml"
BRANCH_GUIDANCE_PATH = (
    ROOT
    / "research_pack"
    / "51_phase_g14_10_branch_protection_guidance.md"
)


class TestCiPolicyWorkflow(unittest.TestCase):
    def test_workflow_contains_required_jobs_and_release_candidate_command(self) -> None:
        self.assertTrue(
            WORKFLOW_PATH.exists(),
            msg=f"missing workflow file: {WORKFLOW_PATH}",
        )
        text = WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("jobs:", text)
        self.assertIn("gate:", text)
        self.assertIn("release-candidate:", text)
        self.assertIn("needs:", text)
        self.assertIn("python -m gf01 gate", text)
        self.assertIn("python -m gf01 release-candidate-check", text)
        self.assertIn("--require-previous-manifest", text)
        self.assertIn("--min-public-novelty-ratio 1.0", text)

    def test_branch_protection_guidance_requires_both_checks(self) -> None:
        self.assertTrue(
            BRANCH_GUIDANCE_PATH.exists(),
            msg=f"missing guidance file: {BRANCH_GUIDANCE_PATH}",
        )
        text = BRANCH_GUIDANCE_PATH.read_text(encoding="utf-8")
        self.assertIn("GF01 Gate / gate", text)
        self.assertIn("GF01 Gate / release-candidate", text)


if __name__ == "__main__":
    unittest.main()
