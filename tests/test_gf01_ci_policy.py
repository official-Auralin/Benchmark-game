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

import re
import unittest
from pathlib import Path

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover - discover mode imports test modules top-level.
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "gf01-gate.yml"
BRANCH_GUIDANCE_PATH = (
    ROOT
    / "research_pack"
    / "51_phase_g14_10_branch_protection_guidance.md"
)

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


class TestCiPolicyWorkflow(unittest.TestCase):
    def _read_workflow_text(self) -> str:
        self.assertTrue(
            WORKFLOW_PATH.exists(),
            msg=f"missing workflow file: {WORKFLOW_PATH}",
        )
        return WORKFLOW_PATH.read_text(encoding="utf-8")

    def _step_block(self, text: str, step_name: str) -> str:
        pattern = re.compile(
            rf"(?ms)^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - name: |\Z)"
        )
        match = pattern.search(text)
        self.assertIsNotNone(match, msg=f"missing workflow step: {step_name}")
        assert match is not None  # narrow type for mypy/linters
        return match.group(0)

    def test_workflow_contains_required_jobs_and_release_candidate_command(self) -> None:
        text = self._read_workflow_text()

        self.assertIn("jobs:", text)
        self.assertIn("gate:", text)
        self.assertIn("release-candidate:", text)
        self.assertIn("needs:", text)
        self.assertIn("python -m gf01 gate", text)
        self.assertIn("python -m gf01 release-candidate-check", text)
        self.assertIn("--require-previous-manifest", text)
        self.assertIn("--min-public-novelty-ratio 1.0", text)

        # Scope workflow-hardening checks to the relevant status-publish steps
        # so unrelated formatting changes elsewhere in the file do not break the
        # test.
        gate_status_step = self._step_block(text, "Publish Gate Status Context")
        rc_status_step = self._step_block(text, "Publish Release Candidate Status Context")
        for step_block in (gate_status_step, rc_status_step):
            self.assertIn("continue-on-error: true", step_block)
            self.assertIn("jq -n", step_block)
            self.assertIn("--retry 3", step_block)

        self.assertIn("STATUS_CONTEXT: GF01 Gate / gate", gate_status_step)
        self.assertIn(
            "STATUS_CONTEXT: GF01 Gate / release-candidate", rc_status_step
        )

    def test_workflow_avoids_private_paths(self) -> None:
        text = self._read_workflow_text()
        for forbidden in ("research_pack/", "readings/"):
            self.assertNotIn(
                forbidden,
                text,
                msg=(
                    "public mirror workflow must not depend on private/local-only "
                    f"paths ({forbidden})"
                ),
            )

    def test_branch_protection_guidance_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                BRANCH_GUIDANCE_PATH.exists(),
                msg=(
                    "public mirror should not include private research_pack "
                    f"artifact: {BRANCH_GUIDANCE_PATH}"
                ),
            )
            return
        self.assertTrue(
            BRANCH_GUIDANCE_PATH.exists(),
            msg=f"missing guidance file in private source repo: {BRANCH_GUIDANCE_PATH}",
        )

    @unittest.skipUnless(
        BRANCH_GUIDANCE_PATH.exists(),
        "branch-protection guidance is absent in this repository scope",
    )
    def test_branch_protection_guidance_requires_both_checks(self) -> None:
        # In the private source repository BRANCH_GUIDANCE_PATH must exist.
        # This content test is expected to skip only in public mirror scope,
        # where private research_pack artifacts are intentionally not mirrored.
        text = BRANCH_GUIDANCE_PATH.read_text(encoding="utf-8")
        self.assertIn("GF01 Gate / gate", text)
        self.assertIn("GF01 Gate / release-candidate", text)


if __name__ == "__main__":
    unittest.main()
