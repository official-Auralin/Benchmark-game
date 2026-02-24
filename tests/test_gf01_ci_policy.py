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
    def _parse_workflow_jobs(self, text: str) -> dict[str, dict[str, object]]:
        """Parse the subset of workflow YAML needed by these regression checks.

        This avoids adding a PyYAML dependency in the test runtime while still
        validating workflow structure (jobs/steps/env/run) instead of relying on
        broad global string fragments.
        """

        def indent_of(line: str) -> int:
            return len(line) - len(line.lstrip(" "))

        def parse_value(raw: str) -> object:
            value = raw.strip()
            if not value:
                return ""
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                return value[1:-1]
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            return value

        jobs: dict[str, dict[str, object]] = {}
        lines = text.splitlines()

        in_jobs = False
        current_job_name: str | None = None
        current_job: dict[str, object] | None = None
        current_step: dict[str, object] | None = None
        current_nested_map_key: str | None = None
        current_nested_map_indent: int | None = None
        collect_job_needs = False
        run_block_key: str | None = None
        run_block_indent: int | None = None
        run_block_lines: list[str] = []

        i = 0
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            indent = indent_of(raw)

            # Continue/finalize multiline step blocks (e.g., run: |).
            if run_block_key is not None and current_step is not None:
                if stripped == "":
                    run_block_lines.append("")
                    i += 1
                    continue
                assert run_block_indent is not None
                if indent >= run_block_indent:
                    run_block_lines.append(raw[run_block_indent:])
                    i += 1
                    continue
                current_step[run_block_key] = "\n".join(run_block_lines)
                run_block_key = None
                run_block_indent = None
                run_block_lines = []
                # Reprocess this line after finalizing the block scalar.
                continue

            if stripped == "" or stripped.startswith("#"):
                i += 1
                continue

            if not in_jobs:
                if indent == 0 and stripped == "jobs:":
                    in_jobs = True
                i += 1
                continue

            # End any nested step map if indentation drops.
            if current_nested_map_key is not None and current_step is not None:
                assert current_nested_map_indent is not None
                if indent < current_nested_map_indent:
                    current_nested_map_key = None
                    current_nested_map_indent = None

            # jobs: child job key (2-space indent)
            if indent == 2 and stripped.endswith(":") and not stripped.startswith("- "):
                current_job_name = stripped[:-1]
                current_job = {"steps": [], "needs": []}
                jobs[current_job_name] = current_job
                current_step = None
                current_nested_map_key = None
                current_nested_map_indent = None
                collect_job_needs = False
                i += 1
                continue

            if current_job is None:
                i += 1
                continue

            # Job-level "needs" list
            if collect_job_needs:
                if indent == 6 and stripped.startswith("- "):
                    needs = current_job.setdefault("needs", [])
                    assert isinstance(needs, list)
                    needs.append(stripped[2:].strip())
                    i += 1
                    continue
                collect_job_needs = False

            # Job-level keys (4-space indent)
            if indent == 4 and ":" in stripped and not stripped.startswith("- "):
                key, value_part = stripped.split(":", 1)
                key = key.strip()
                value_part = value_part.strip()
                if key == "needs":
                    if value_part:
                        current_job["needs"] = [str(parse_value(value_part))]
                    else:
                        current_job["needs"] = []
                        collect_job_needs = True
                    i += 1
                    continue
                if key == "steps":
                    i += 1
                    continue
                current_job[key] = parse_value(value_part)
                i += 1
                continue

            # Step start (6-space indent, list item)
            if indent == 6 and stripped.startswith("- "):
                current_step = {}
                current_nested_map_key = None
                current_nested_map_indent = None
                item_body = stripped[2:].strip()
                if ":" in item_body:
                    key, value_part = item_body.split(":", 1)
                    key = key.strip()
                    value_part = value_part.strip()
                    current_step[key] = parse_value(value_part)
                steps = current_job.setdefault("steps", [])
                assert isinstance(steps, list)
                steps.append(current_step)
                i += 1
                continue

            # Step-level nested maps like env:/with:
            if (
                current_step is not None
                and current_nested_map_key is not None
                and current_nested_map_indent is not None
                and indent >= current_nested_map_indent
                and ":" in stripped
                and not stripped.startswith("- ")
            ):
                key, value_part = stripped.split(":", 1)
                nested = current_step.setdefault(current_nested_map_key, {})
                assert isinstance(nested, dict)
                nested[key.strip()] = parse_value(value_part.strip())
                i += 1
                continue

            # Step-level keys (8-space indent)
            if current_step is not None and indent == 8 and ":" in stripped:
                key, value_part = stripped.split(":", 1)
                key = key.strip()
                value_part = value_part.strip()

                if value_part in {"|", ">"}:
                    run_block_key = key
                    run_block_indent = 10
                    run_block_lines = []
                    i += 1
                    continue

                if value_part == "":
                    current_step[key] = {}
                    current_nested_map_key = key
                    current_nested_map_indent = 10
                    i += 1
                    continue

                current_step[key] = parse_value(value_part)
                current_nested_map_key = None
                current_nested_map_indent = None
                i += 1
                continue

            i += 1

        # Finalize trailing multiline block if file ends inside it.
        if run_block_key is not None and current_step is not None:
            current_step[run_block_key] = "\n".join(run_block_lines)

        return jobs

    def _read_workflow_text(self) -> str:
        self.assertTrue(
            WORKFLOW_PATH.exists(),
            msg=f"missing workflow file: {WORKFLOW_PATH}",
        )
        return WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_contains_required_jobs_and_release_candidate_command(self) -> None:
        text = self._read_workflow_text()
        jobs = self._parse_workflow_jobs(text)
        self.assertIn("gate", jobs)
        self.assertIn("release-candidate", jobs)

        rc_needs = jobs["release-candidate"].get("needs", [])
        self.assertIsInstance(rc_needs, list)
        self.assertIn("gate", rc_needs)

        def steps_by_name(job_name: str) -> dict[str, dict[str, object]]:
            job = jobs[job_name]
            steps = job.get("steps", [])
            self.assertIsInstance(steps, list, msg=f"{job_name} steps must be a list")
            by_name: dict[str, dict[str, object]] = {}
            for step in steps:
                self.assertIsInstance(step, dict)
                name = step.get("name")
                if isinstance(name, str):
                    by_name[name] = step
            return by_name

        gate_steps = steps_by_name("gate")
        rc_steps = steps_by_name("release-candidate")

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
            env = step.get("env", {})
            self.assertIsInstance(env, dict)

        self.assertEqual(
            gate_status_step.get("env", {}).get("STATUS_CONTEXT"),
            "GF01 Gate / gate",
        )
        self.assertEqual(
            rc_status_step.get("env", {}).get("STATUS_CONTEXT"),
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


if __name__ == "__main__":
    unittest.main()
