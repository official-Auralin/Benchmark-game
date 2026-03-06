"""
Regression tests for pilot campaign analysis and calibration-trigger checks.

These tests verify that the analysis command consumes campaign artifacts,
produces a machine-checkable summary, and fails clearly when required inputs
are missing.
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

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from .workflow_artifact_harness import clone_tree, run_cli
except ImportError:  # pragma: no cover
    from workflow_artifact_harness import clone_tree, run_cli


class TestPilotAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._prepared_root = Path(tempfile.mkdtemp(prefix="gf01-analysis-fixtures-"))
        cls.addClassCleanup(shutil.rmtree, cls._prepared_root, ignore_errors=True)

        cls.standard_freeze_dir = cls._prepared_root / "standard_freeze"
        cls.standard_run_dir = cls._prepared_root / "standard_runs"
        standard_freeze = run_cli(
            [
                "freeze-pilot",
                "--freeze-id",
                "gf01-pilot-freeze-analysis-test",
                "--split",
                "pilot_internal_test",
                "--seed-start",
                "9400",
                "--count",
                "8",
                "--out-dir",
                str(cls.standard_freeze_dir),
            ]
        )
        if standard_freeze.returncode != 0:
            raise AssertionError(standard_freeze.stdout + standard_freeze.stderr)

        standard_campaign = run_cli(
            [
                "pilot-campaign",
                "--freeze-dir",
                str(cls.standard_freeze_dir),
                "--out-dir",
                str(cls.standard_run_dir),
                "--baseline-panel",
                "random,greedy,oracle",
                "--baseline-policy-level",
                "core",
                "--renderer-track",
                "json",
                "--seed",
                "77",
            ]
        )
        if standard_campaign.returncode != 0:
            raise AssertionError(standard_campaign.stdout + standard_campaign.stderr)

        cls.legacy_freeze_dir = cls._prepared_root / "legacy_freeze"
        cls.legacy_run_dir = cls._prepared_root / "legacy_runs"
        legacy_freeze = run_cli(
            [
                "freeze-pilot",
                "--freeze-id",
                "gf01-pilot-freeze-analysis-legacy-test",
                "--split",
                "pilot_internal_test",
                "--seed-start",
                "9500",
                "--count",
                "6",
                "--out-dir",
                str(cls.legacy_freeze_dir),
            ]
        )
        if legacy_freeze.returncode != 0:
            raise AssertionError(legacy_freeze.stdout + legacy_freeze.stderr)

        legacy_campaign = run_cli(
            [
                "pilot-campaign",
                "--freeze-dir",
                str(cls.legacy_freeze_dir),
                "--out-dir",
                str(cls.legacy_run_dir),
                "--baseline-panel",
                "random,greedy,oracle",
                "--baseline-policy-level",
                "core",
                "--renderer-track",
                "json",
                "--seed",
                "91",
            ]
        )
        if legacy_campaign.returncode != 0:
            raise AssertionError(legacy_campaign.stdout + legacy_campaign.stderr)

    def test_pilot_analysis_emits_trigger_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-analysis-") as tmp:
            run_dir = clone_tree(self.standard_run_dir, Path(tmp) / "runs")

            analysis = run_cli(
                [
                    "pilot-analyze",
                    "--campaign-dir",
                    str(run_dir),
                    "--eval-track",
                    "EVAL-CB",
                    "--mode",
                    "normal",
                ]
            )
            self.assertEqual(analysis.returncode, 0, msg=analysis.stdout + analysis.stderr)
            payload = json.loads(analysis.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("policy_reference"), "DEC-014d")
            self.assertEqual(payload.get("complexity_policy_version"), "gf01.complexity_policy.v1")
            self.assertIn("discrimination_check", payload)
            self.assertIn("shortcut_check", payload)
            self.assertIn("complexity_diagnostics", payload)
            self.assertIn("sample_progress", payload)
            self.assertTrue(Path(payload["out_path"]).exists())
            self.assertIn(payload.get("recommendation"), {"keep_coefficients", "recalibrate_normal_window"})

    def test_pilot_analysis_migrates_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-analysis-") as tmp:
            run_dir = clone_tree(self.legacy_run_dir, Path(tmp) / "runs")

            runs_path = run_dir / "runs_combined.jsonl"
            rows = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
            for row in rows:
                row.pop("renderer_policy_version", None)
                row.pop("renderer_profile_id", None)
                row.pop("adaptation_policy_version", None)
                row.pop("adaptation_condition", None)
                row.pop("adaptation_budget_tokens", None)
                row.pop("adaptation_data_scope", None)
                row.pop("adaptation_protocol_id", None)
            runs_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

            analysis = run_cli(
                [
                    "pilot-analyze",
                    "--campaign-dir",
                    str(run_dir),
                    "--eval-track",
                    "EVAL-CB",
                    "--mode",
                    "normal",
                ]
            )
            self.assertEqual(analysis.returncode, 0, msg=analysis.stdout + analysis.stderr)
            payload = json.loads(analysis.stdout)
            legacy = payload.get("legacy_migration", {})
            self.assertEqual(payload.get("status"), "ok")
            self.assertTrue(bool(legacy.get("applied", False)))
            self.assertIn("report", legacy)

    def test_pilot_analysis_requires_campaign_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-analysis-") as tmp:
            campaign_dir = Path(tmp) / "campaign"
            campaign_dir.mkdir(parents=True, exist_ok=True)
            proc = run_cli(
                [
                    "pilot-analyze",
                    "--campaign-dir",
                    str(campaign_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "campaign_runs_missing")


if __name__ == "__main__":
    unittest.main()
