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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestPilotAnalysis(unittest.TestCase):
    def test_pilot_analysis_emits_trigger_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-analysis-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            run_dir = Path(tmp) / "runs"

            freeze = _run_cli(
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
                    str(freeze_dir),
                ]
            )
            self.assertEqual(freeze.returncode, 0, msg=freeze.stdout + freeze.stderr)

            campaign = _run_cli(
                [
                    "pilot-campaign",
                    "--freeze-dir",
                    str(freeze_dir),
                    "--out-dir",
                    str(run_dir),
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
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)

            analysis = _run_cli(
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
            self.assertIn("discrimination_check", payload)
            self.assertIn("shortcut_check", payload)
            self.assertIn("sample_progress", payload)
            self.assertTrue(Path(payload["out_path"]).exists())
            self.assertIn(payload.get("recommendation"), {"keep_coefficients", "recalibrate_normal_window"})

    def test_pilot_analysis_requires_campaign_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-analysis-") as tmp:
            campaign_dir = Path(tmp) / "campaign"
            campaign_dir.mkdir(parents=True, exist_ok=True)
            proc = _run_cli(
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
