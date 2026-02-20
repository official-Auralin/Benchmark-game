"""
Regression tests for release report/panel policy checks.

These tests ensure release artifacts satisfy baseline-panel coverage and
track/slice reporting requirements before publication usage.
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


class TestReleaseReportPolicy(unittest.TestCase):
    def test_release_report_check_passes_for_full_panel_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-report-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            campaign_dir = Path(tmp) / "campaign"

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-report-full",
                    "--split",
                    "pilot_internal_release_report",
                    "--seed-start",
                    "9500",
                    "--count",
                    "2",
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
                    str(campaign_dir),
                    "--baseline-panel",
                    "random,greedy,search,tool,oracle",
                    "--baseline-policy-level",
                    "full",
                    "--renderer-track",
                    "json",
                    "--seed",
                    "55",
                ]
            )
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)

            check = _run_cli(
                [
                    "release-report-check",
                    "--runs",
                    str(payload["runs_path"]),
                    "--manifest",
                    str(freeze_dir / "split_manifest_v1.json"),
                    "--baseline-policy-level",
                    "full",
                ]
            )
            self.assertEqual(check.returncode, 0, msg=check.stdout + check.stderr)
            check_payload = json.loads(check.stdout)
            self.assertEqual(check_payload.get("status"), "ok")
            self.assertTrue(check_payload.get("passed"))

    def test_release_report_check_fails_when_full_policy_missing_agents(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-report-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            campaign_dir = Path(tmp) / "campaign"

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-report-core",
                    "--split",
                    "pilot_internal_release_report",
                    "--seed-start",
                    "9500",
                    "--count",
                    "2",
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
                    str(campaign_dir),
                    "--baseline-panel",
                    "random,greedy,oracle",
                    "--baseline-policy-level",
                    "core",
                    "--renderer-track",
                    "json",
                    "--seed",
                    "55",
                ]
            )
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)

            check = _run_cli(
                [
                    "release-report-check",
                    "--runs",
                    str(payload["runs_path"]),
                    "--manifest",
                    str(freeze_dir / "split_manifest_v1.json"),
                    "--baseline-policy-level",
                    "full",
                ]
            )
            self.assertEqual(check.returncode, 2, msg=check.stdout + check.stderr)
            check_payload = json.loads(check.stdout)
            self.assertEqual(check_payload.get("error_type"), "release_report_policy_violation")
            missing_agents = set(check_payload.get("missing_required_agents", []))
            self.assertIn("search", missing_agents)
            self.assertIn("tool", missing_agents)

    def test_release_report_check_passes_for_core_policy_on_core_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-report-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            campaign_dir = Path(tmp) / "campaign"

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-report-core-pass",
                    "--split",
                    "pilot_internal_release_report",
                    "--seed-start",
                    "9500",
                    "--count",
                    "2",
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
                    str(campaign_dir),
                    "--baseline-panel",
                    "random,greedy,oracle",
                    "--baseline-policy-level",
                    "core",
                    "--renderer-track",
                    "json",
                    "--seed",
                    "55",
                ]
            )
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)

            check = _run_cli(
                [
                    "release-report-check",
                    "--runs",
                    str(payload["runs_path"]),
                    "--manifest",
                    str(freeze_dir / "split_manifest_v1.json"),
                    "--baseline-policy-level",
                    "core",
                ]
            )
            self.assertEqual(check.returncode, 0, msg=check.stdout + check.stderr)
            check_payload = json.loads(check.stdout)
            self.assertEqual(check_payload.get("status"), "ok")
            self.assertTrue(check_payload.get("passed"))


if __name__ == "__main__":
    unittest.main()
