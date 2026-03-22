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
__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

import json
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
    / "docs"
    / "PUBLIC_MIRROR.md"
)
CONTRACT_INVENTORY_PATH = ROOT / "spec" / "contract_inventory.json"

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


class TestCiPolicyWorkflow(unittest.TestCase):
    def _contract_inventory(self) -> dict[str, object]:
        self.assertTrue(
            CONTRACT_INVENTORY_PATH.exists(),
            msg=f"missing contract inventory: {CONTRACT_INVENTORY_PATH}",
        )
        payload = json.loads(CONTRACT_INVENTORY_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(payload, dict)
        return payload

    def _read_workflow_text(self) -> str:
        self.assertTrue(
            WORKFLOW_PATH.exists(),
            msg=f"missing workflow file: {WORKFLOW_PATH}",
        )
        return WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_contains_required_jobs_and_release_candidate_command(self) -> None:
        text = self._read_workflow_text()
        jobs = parse_workflow_jobs_subset(text)
        self.assertIn("core-gate", jobs)
        self.assertIn("spec-build", jobs)
        self.assertIn("human-ui", jobs)
        self.assertIn("type-and-coverage", jobs)
        self.assertIn("gate", jobs)
        self.assertIn("release-candidate", jobs)

        gate_needs = job_needs(jobs["gate"])
        self.assertIn("core-gate", gate_needs)
        self.assertIn("spec-build", gate_needs)
        self.assertIn("human-ui", gate_needs)
        self.assertIn("type-and-coverage", gate_needs)

        rc_needs = job_needs(jobs["release-candidate"])
        self.assertIn("gate", rc_needs)

        core_steps = steps_by_name(jobs["core-gate"])
        spec_steps = steps_by_name(jobs["spec-build"])
        ui_steps = steps_by_name(jobs["human-ui"])
        type_steps = steps_by_name(jobs["type-and-coverage"])
        gate_steps = steps_by_name(jobs["gate"])
        rc_steps = steps_by_name(jobs["release-candidate"])

        self.assertIn("Run GF01 Core Gate", core_steps)
        self.assertIn("Run Spec Build Check", spec_steps)
        self.assertIn("Run Human UI Smoke And Tests", ui_steps)
        self.assertIn("Run Targeted Mypy", type_steps)
        self.assertIn("Run Coverage Suite", type_steps)
        self.assertIn("Run Integrated Release Candidate Check", rc_steps)
        self.assertIn("Publish Gate Status Context", gate_steps)
        self.assertIn("Publish Release Candidate Status Context", rc_steps)

        core_run = str(core_steps["Run GF01 Core Gate"].get("run", ""))
        self.assertIn("python -m gf01 gate", core_run)
        self.assertIn("--unittest-shards 2", core_run)

        spec_run = str(spec_steps["Run Spec Build Check"].get("run", ""))
        self.assertIn("python scripts/build_spec.py --check", spec_run)
        self.assertIn(
            "python -m unittest tests.test_docs_spec_sync tests.test_spec_tex_integrity -v",
            spec_run,
        )

        ui_run = str(ui_steps["Run Human UI Smoke And Tests"].get("run", ""))
        self.assertIn("tests.test_gf01_play_loop", ui_run)
        self.assertIn("tests.test_r1_renderer_modules", ui_run)
        self.assertIn("SDL_VIDEODRIVER=dummy", ui_run)
        self.assertEqual(
            step_env(ui_steps["Run Human UI Smoke And Tests"]).get("GF01_REQUIRE_PYGAME"),
            "1",
        )
        self.assertEqual(
            step_env(ui_steps["Run Human UI Smoke And Tests"]).get("SDL_AUDIODRIVER"),
            "dummy",
        )

        mypy_run = str(type_steps["Run Targeted Mypy"].get("run", ""))
        self.assertIn("python -m mypy", mypy_run)
        self.assertIn("gf01/generator.py", mypy_run)

        coverage_run = str(type_steps["Run Coverage Suite"].get("run", ""))
        self.assertIn("python -m coverage run", coverage_run)
        self.assertIn("python -m coverage report", coverage_run)

        rc_run = str(rc_steps["Run Integrated Release Candidate Check"].get("run", ""))
        self.assertIn("python -m gf01 release-candidate-check", rc_run)
        self.assertIn("--require-previous-manifest", rc_run)
        self.assertIn("--min-public-novelty-ratio 1.0", rc_run)

        inventory = self._contract_inventory()
        expected_contexts = inventory.get("ci_status_contexts", [])
        self.assertEqual(
            expected_contexts,
            ["GF01 Gate / gate", "GF01 Gate / release-candidate"],
        )

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
            [
                step_env(gate_status_step).get("STATUS_CONTEXT"),
                step_env(rc_status_step).get("STATUS_CONTEXT"),
            ],
            expected_contexts,
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
        self.assertTrue(
            BRANCH_GUIDANCE_PATH.exists(),
            msg=f"missing public-mirror guidance doc: {BRANCH_GUIDANCE_PATH}",
        )

    @unittest.skipUnless(
        BRANCH_GUIDANCE_PATH.exists(),
        "branch-protection guidance is absent in this repository scope",
    )
    def test_branch_protection_guidance_requires_both_checks(self) -> None:
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
