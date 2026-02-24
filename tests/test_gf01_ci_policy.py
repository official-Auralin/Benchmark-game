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

try:
    from .repo_scope import is_public_mirror
    from .workflow_parser_subset import (
        WorkflowSubsetParseError,
        job_needs,
        job_steps,
        parse_workflow_jobs_subset,
        step_env,
        steps_by_name,
    )
except ImportError:  # pragma: no cover - discover mode imports test modules top-level.
    from repo_scope import is_public_mirror
    from workflow_parser_subset import (
        WorkflowSubsetParseError,
        job_needs,
        job_steps,
        parse_workflow_jobs_subset,
        step_env,
        steps_by_name,
    )


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

    def test_workflow_contains_required_jobs_and_release_candidate_command(self) -> None:
        text = self._read_workflow_text()
        jobs = parse_workflow_jobs_subset(text)
        self.assertIn("gate", jobs)
        self.assertIn("release-candidate", jobs)

        rc_needs = job_needs(jobs["release-candidate"])
        self.assertIn("gate", rc_needs)

        gate_steps = steps_by_name(jobs["gate"])
        rc_steps = steps_by_name(jobs["release-candidate"])

        self.assertIn("Run GF01 Gate", gate_steps)
        self.assertIn("Run Integrated Release Candidate Check", rc_steps)
        self.assertIn("Publish Gate Status Context", gate_steps)
        self.assertIn("Publish Release Candidate Status Context", rc_steps)

        gate_run = str(gate_steps["Run GF01 Gate"].get("run", ""))
        self.assertIn("python -m gf01 gate", gate_run)

        rc_run = str(rc_steps["Run Integrated Release Candidate Check"].get("run", ""))
        self.assertIn("python -m gf01 release-candidate-check", rc_run)
        self.assertIn("--require-previous-manifest", rc_run)
        self.assertIn("--min-public-novelty-ratio 1.0", rc_run)

        gate_status_step = gate_steps["Publish Gate Status Context"]
        rc_status_step = rc_steps["Publish Release Candidate Status Context"]
        for step in (gate_status_step, rc_status_step):
            self.assertTrue(
                step.get("continue-on-error") is True,
                msg="status-publish step must be non-blocking (continue-on-error: true)",
            )
            run = str(step.get("run", ""))
            self.assertIn("jq -n", run)
            self.assertIn("--retry 3", run)
            step_env(step)

        self.assertEqual(
            step_env(gate_status_step).get("STATUS_CONTEXT"),
            "GF01 Gate / gate",
        )
        self.assertEqual(
            step_env(rc_status_step).get("STATUS_CONTEXT"),
            "GF01 Gate / release-candidate",
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

class TestParseWorkflowJobsUnit(unittest.TestCase):
    """Focused unit tests for ``parse_workflow_jobs_subset``."""

    def _parse(self, yaml_text: str) -> dict[str, dict[str, object]]:
        return parse_workflow_jobs_subset(yaml_text)

    def test_multiple_needs_inline_and_list(self) -> None:
        yaml_text = """
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo build

  test:
    needs: [build, lint]
    runs-on: ubuntu-latest
    steps:
      - run: echo test

  deploy:
    needs:
      - build
      - test
    runs-on: ubuntu-latest
    steps:
      - run: echo deploy
"""
        jobs = self._parse(yaml_text)
        self.assertIn("build", jobs)
        self.assertIn("test", jobs)
        self.assertIn("deploy", jobs)
        self.assertCountEqual(job_needs(jobs["test"]), ["build", "lint"])
        self.assertCountEqual(job_needs(jobs["deploy"]), ["build", "test"])

    def test_nested_env_and_with_at_varying_indents(self) -> None:
        yaml_text = """
jobs:
  complex:
    runs-on: ubuntu-latest
    env:
      TOP_LEVEL: 1
    steps:
      - name: step-with-env-and-with
        env:
          STEP_LEVEL: 2
        with:
          some-input: value
        run: echo "$TOP_LEVEL $STEP_LEVEL"

      - name: step-with-only-env
        env:
          ONLY_ENV: 3
        run: echo "$ONLY_ENV"

      - name: step-with-only-with
        with:
          another-input: other
        run: echo "with only"
"""
        jobs = self._parse(yaml_text)
        complex_job = jobs["complex"]
        self.assertEqual(complex_job.get("env", {}).get("TOP_LEVEL"), "1")

        steps = job_steps(complex_job)
        self.assertGreaterEqual(len(steps), 3)

        step0 = steps[0]
        self.assertIn("env", step0)
        self.assertIn("with", step0)

        step1 = steps[1]
        self.assertIn("env", step1)
        self.assertNotIn("with", step1)

        step2 = steps[2]
        self.assertNotIn("env", step2)
        self.assertIn("with", step2)

    def test_multiline_run_block_literal_with_trailing_blank_lines(self) -> None:
        yaml_text = """
jobs:
  multiline_literal:
    runs-on: ubuntu-latest
    steps:
      - name: literal-block
        run: |
          echo "line1"
          echo "line2"

          echo "line4"

"""
        jobs = self._parse(yaml_text)
        run_script = str(job_steps(jobs["multiline_literal"])[0]["run"])
        self.assertIn('echo "line1"', run_script)
        self.assertIn('echo "line2"', run_script)
        self.assertIn('echo "line4"', run_script)
        self.assertTrue(run_script.strip().endswith('echo "line4"'))

    def test_multiline_run_block_folded_with_trailing_blank_lines(self) -> None:
        yaml_text = """
jobs:
  multiline_folded:
    runs-on: ubuntu-latest
    steps:
      - name: folded-block
        run: >
          echo "line1"
          echo "line2"

          echo "line4"

"""
        jobs = self._parse(yaml_text)
        run_script = str(job_steps(jobs["multiline_folded"])[0]["run"])
        self.assertIn('echo "line1"', run_script)
        self.assertIn('echo "line2"', run_script)
        self.assertIn('echo "line4"', run_script)

    def test_nested_with_multiline_block_scalar(self) -> None:
        yaml_text = """
jobs:
  artifact:
    runs-on: ubuntu-latest
    steps:
      - name: upload
        uses: actions/upload-artifact@v4
        with:
          name: artifact-name
          path: |
            one.txt
            two.txt

            three.txt
"""
        jobs = self._parse(yaml_text)
        step = job_steps(jobs["artifact"])[0]
        with_map = step.get("with", {})
        self.assertIsInstance(with_map, dict)
        self.assertEqual(with_map.get("name"), "artifact-name")
        path_value = str(with_map.get("path", ""))
        self.assertIn("one.txt", path_value)
        self.assertIn("two.txt", path_value)
        self.assertIn("three.txt", path_value)

    def test_file_ending_inside_multiline_block(self) -> None:
        yaml_text = """
jobs:
  end_in_block:
    runs-on: ubuntu-latest
    steps:
      - name: end-inside-block
        run: |
          echo "last line"
"""
        jobs = self._parse(yaml_text)
        run_script = str(job_steps(jobs["end_in_block"])[0]["run"])
        self.assertIn('echo "last line"', run_script)
        self.assertTrue(run_script.strip().endswith('echo "last line"'))

    def test_raises_on_invalid_odd_indentation(self) -> None:
        yaml_text = """
jobs:
  bad:
    runs-on: ubuntu-latest
    steps:
      - name: bad-indent
         run: echo nope
"""
        with self.assertRaises(WorkflowSubsetParseError):
            self._parse(yaml_text)


if __name__ == "__main__":
    unittest.main()
