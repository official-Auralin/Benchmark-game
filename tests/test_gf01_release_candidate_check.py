"""
Regression tests for the integrated release-candidate check command.

These tests verify deterministic composition of governance, report, and
package stages under one machine-checkable command path.
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


def _remap_freeze_to_official_splits(freeze_dir: Path) -> None:
    bundle_path = freeze_dir / "instance_bundle_v1.json"
    manifest_path = freeze_dir / "split_manifest_v1.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = bundle.get("instances", [])
    if not isinstance(instances, list) or len(instances) != 3:
        raise AssertionError("expected exactly 3 instances for deterministic remap")

    split_order = [
        "public_dev",
        "public_val",
        "private_eval",
    ]
    for instance, split_id in zip(instances, split_order):
        if not isinstance(instance, dict):
            raise AssertionError("instance row is not an object")
        instance["split_id"] = split_id

    manifest_rows = []
    for instance in instances:
        manifest_rows.append(
            {
                "instance_id": str(instance.get("instance_id", "")),
                "split_id": str(instance.get("split_id", "")),
                "mode": str(instance.get("mode", "normal")),
                "seed": int(instance.get("seed", 0)),
                "t_star": int(instance.get("t_star", 0)),
                "window_size": int(instance.get("window_size", 0)),
                "budget_timestep": int(instance.get("budget_timestep", 0)),
                "budget_atoms": int(instance.get("budget_atoms", 0)),
            }
        )
    manifest["instance_count"] = len(manifest_rows)
    manifest["instances"] = manifest_rows

    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class TestReleaseCandidateCheck(unittest.TestCase):
    def test_release_candidate_check_passes_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-candidate-") as tmp:
            base = Path(tmp)
            current_freeze = base / "current_freeze"
            previous_freeze = base / "previous_freeze"
            campaign_dir = base / "campaign"
            package_dir = base / "package"

            current = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-candidate-current",
                    "--split",
                    "pilot_internal_release_candidate",
                    "--seed-start",
                    "9600",
                    "--count",
                    "3",
                    "--mode",
                    "normal",
                    "--out-dir",
                    str(current_freeze),
                ]
            )
            self.assertEqual(current.returncode, 0, msg=current.stdout + current.stderr)

            previous = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-candidate-previous",
                    "--split",
                    "pilot_internal_release_candidate_prev",
                    "--seed-start",
                    "9700",
                    "--count",
                    "3",
                    "--mode",
                    "normal",
                    "--out-dir",
                    str(previous_freeze),
                ]
            )
            self.assertEqual(previous.returncode, 0, msg=previous.stdout + previous.stderr)

            _remap_freeze_to_official_splits(current_freeze)
            _remap_freeze_to_official_splits(previous_freeze)

            campaign = _run_cli(
                [
                    "pilot-campaign",
                    "--freeze-dir",
                    str(current_freeze),
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

            check = _run_cli(
                [
                    "release-candidate-check",
                    "--freeze-dir",
                    str(current_freeze),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--previous-manifest",
                    str(previous_freeze / "split_manifest_v1.json"),
                    "--require-previous-manifest",
                    "--min-private-eval-count",
                    "1",
                    "--target-ratios",
                    "public_dev=0.3333333333,public_val=0.3333333333,private_eval=0.3333333333",
                    "--min-public-novelty-ratio",
                    "1.0",
                    "--package-out-dir",
                    str(package_dir),
                ]
            )
            self.assertEqual(check.returncode, 0, msg=check.stdout + check.stderr)
            payload = json.loads(check.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertTrue(payload.get("passed"))
            stages = payload.get("stages", {})
            self.assertTrue(stages.get("release_governance", {}).get("passed"))
            self.assertTrue(stages.get("release_report", {}).get("passed"))
            self.assertTrue(stages.get("release_package", {}).get("passed"))
            self.assertTrue((package_dir / "release_package_manifest.json").exists())

    def test_release_candidate_check_fails_when_release_report_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-candidate-") as tmp:
            base = Path(tmp)
            current_freeze = base / "current_freeze"
            campaign_dir = base / "campaign"

            current = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-candidate-core",
                    "--split",
                    "pilot_internal_release_candidate",
                    "--seed-start",
                    "9800",
                    "--count",
                    "3",
                    "--mode",
                    "normal",
                    "--out-dir",
                    str(current_freeze),
                ]
            )
            self.assertEqual(current.returncode, 0, msg=current.stdout + current.stderr)
            _remap_freeze_to_official_splits(current_freeze)

            campaign = _run_cli(
                [
                    "pilot-campaign",
                    "--freeze-dir",
                    str(current_freeze),
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

            check = _run_cli(
                [
                    "release-candidate-check",
                    "--freeze-dir",
                    str(current_freeze),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--min-private-eval-count",
                    "1",
                    "--target-ratios",
                    "public_dev=0.3333333333,public_val=0.3333333333,private_eval=0.3333333333",
                    "--baseline-policy-level",
                    "full",
                    "--skip-package",
                ]
            )
            self.assertEqual(check.returncode, 2, msg=check.stdout + check.stderr)
            payload = json.loads(check.stdout)
            self.assertEqual(payload.get("error_type"), "release_candidate_check_failed")
            stages = payload.get("stages", {})
            self.assertTrue(stages.get("release_governance", {}).get("passed"))
            self.assertFalse(stages.get("release_report", {}).get("passed"))


if __name__ == "__main__":
    unittest.main()
