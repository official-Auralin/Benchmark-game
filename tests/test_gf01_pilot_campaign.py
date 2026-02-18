"""
Regression tests for pilot campaign execution on frozen packs.

These tests ensure the campaign command emits strict/official artifacts and
enforces output-directory safety behavior.
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


class TestPilotCampaign(unittest.TestCase):
    def test_campaign_writes_official_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-campaign-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            run_dir = Path(tmp) / "runs"
            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-pilot-freeze-campaign-test",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
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
                    str(run_dir),
                    "--baseline-panel",
                    "random,greedy,search,tool,oracle",
                    "--renderer-track",
                    "json",
                    "--seed",
                    "55",
                ]
            )
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)
            summary = json.loads(campaign.stdout)
            self.assertEqual(summary.get("status"), "ok")
            self.assertEqual(summary.get("row_count"), 10)
            self.assertEqual(summary.get("baseline_policy_version"), "gf01.baseline_panel_policy.v1")
            self.assertEqual(summary.get("baseline_policy_level"), "full")
            self.assertEqual(summary.get("renderer_policy_version"), "gf01.renderer_policy.v1")
            self.assertEqual(summary.get("renderer_profile_id"), "canonical-json-v1")

            runs_path = Path(summary["runs_path"])
            val_path = Path(summary["validation_path"])
            report_path = Path(summary["report_path"])
            self.assertTrue(runs_path.exists())
            self.assertTrue(val_path.exists())
            self.assertTrue(report_path.exists())

            rows = []
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            self.assertTrue(rows)
            ta_rows = [row for row in rows if row.get("eval_track") == "EVAL-TA"]
            oc_rows = [row for row in rows if row.get("eval_track") == "EVAL-OC"]
            self.assertTrue(ta_rows)
            self.assertTrue(oc_rows)
            self.assertTrue(
                all(row.get("tool_allowlist_id") == "local-planner-v1" for row in ta_rows)
            )
            self.assertTrue(
                all(row.get("tool_allowlist_id") == "oracle-exact-search-v1" for row in oc_rows)
            )
            self.assertTrue(all(bool(str(row.get("tool_log_hash", "")).strip()) for row in ta_rows))
            self.assertTrue(all(bool(str(row.get("tool_log_hash", "")).strip()) for row in oc_rows))
            self.assertTrue(all(row.get("renderer_policy_version") == "gf01.renderer_policy.v1" for row in rows))
            self.assertTrue(all(row.get("renderer_profile_id") == "canonical-json-v1" for row in rows))
            self.assertTrue(all(row.get("adaptation_condition") == "no_adaptation" for row in rows))
            self.assertTrue(all(int(row.get("adaptation_budget_tokens", -1)) == 0 for row in rows))
            self.assertTrue(all(row.get("adaptation_data_scope") == "none" for row in rows))
            self.assertTrue(all(row.get("adaptation_protocol_id") == "none" for row in rows))

            val_payload = json.loads(val_path.read_text(encoding="utf-8"))
            self.assertEqual(val_payload.get("status"), "ok")
            self.assertTrue(val_payload.get("official_mode"))

    def test_campaign_requires_force_for_non_empty_output_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-campaign-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            run_dir = Path(tmp) / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "existing.txt").write_text("x", encoding="utf-8")

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-pilot-freeze-campaign-test",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
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
                    str(run_dir),
                    "--baseline-panel",
                    "greedy",
                ]
            )
            self.assertEqual(campaign.returncode, 2, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)
            self.assertEqual(payload.get("error_type"), "output_dir_not_empty")

    def test_campaign_rejects_panel_missing_required_full_policy_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-campaign-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            run_dir = Path(tmp) / "runs"
            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-pilot-freeze-campaign-test",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
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
                    str(run_dir),
                    "--baseline-panel",
                    "random,greedy,oracle",
                ]
            )
            self.assertEqual(campaign.returncode, 2, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)
            self.assertEqual(payload.get("error_type"), "baseline_panel_policy_violation")

    def test_campaign_core_policy_accepts_core_panel(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-campaign-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            run_dir = Path(tmp) / "runs"
            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-pilot-freeze-campaign-test",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
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
                    str(run_dir),
                    "--baseline-panel",
                    "random,greedy,oracle",
                    "--baseline-policy-level",
                    "core",
                ]
            )
            self.assertEqual(campaign.returncode, 0, msg=campaign.stdout + campaign.stderr)
            payload = json.loads(campaign.stdout)
            self.assertEqual(payload.get("baseline_policy_level"), "core")
            self.assertEqual(payload.get("row_count"), 6)


if __name__ == "__main__":
    unittest.main()
