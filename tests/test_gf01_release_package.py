"""
Regression tests for reproducibility package assembly.

These tests verify that `release-package` validates source artifacts and emits
deterministic package files for reruns.
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


class TestReleasePackage(unittest.TestCase):
    def test_release_package_emits_manifest_and_instructions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-package-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            campaign_dir = Path(tmp) / "campaign"
            package_dir = Path(tmp) / "package"

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-package-test",
                    "--split",
                    "pilot_internal_release_test",
                    "--seed-start",
                    "9400",
                    "--count",
                    "1",
                    "--mode",
                    "normal",
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

            analysis = _run_cli(
                [
                    "pilot-analyze",
                    "--campaign-dir",
                    str(campaign_dir),
                    "--eval-track",
                    "EVAL-CB",
                    "--mode",
                    "normal",
                ]
            )
            self.assertEqual(analysis.returncode, 0, msg=analysis.stdout + analysis.stderr)

            package = _run_cli(
                [
                    "release-package",
                    "--freeze-dir",
                    str(freeze_dir),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--out-dir",
                    str(package_dir),
                ]
            )
            self.assertEqual(package.returncode, 0, msg=package.stdout + package.stderr)
            payload = json.loads(package.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("schema_version"), "gf01.release_package.v1")

            manifest_path = Path(payload["manifest_path"])
            instructions_path = Path(payload["instructions_path"])
            artifacts_dir = package_dir / "artifacts"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(instructions_path.exists())
            self.assertTrue(artifacts_dir.exists())
            self.assertTrue((artifacts_dir / "instance_bundle_v1.json").exists())
            self.assertTrue((artifacts_dir / "split_manifest_v1.json").exists())
            self.assertTrue((artifacts_dir / "runs_combined.jsonl").exists())
            self.assertTrue((artifacts_dir / "official_validation.json").exists())
            self.assertTrue((artifacts_dir / "official_report.json").exists())
            self.assertTrue((artifacts_dir / "pilot_analysis.json").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("schema_version"), "gf01.release_package.v1")
            self.assertEqual(manifest.get("family_id"), "GF-01")
            self.assertTrue(manifest.get("files"))

    def test_release_package_requires_force_for_non_empty_output_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-package-") as tmp:
            freeze_dir = Path(tmp) / "freeze"
            campaign_dir = Path(tmp) / "campaign"
            package_dir = Path(tmp) / "package"
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "existing.txt").write_text("x", encoding="utf-8")

            freeze = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-release-package-test",
                    "--split",
                    "pilot_internal_release_test",
                    "--seed-start",
                    "9400",
                    "--count",
                    "1",
                    "--mode",
                    "normal",
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

            package = _run_cli(
                [
                    "release-package",
                    "--freeze-dir",
                    str(freeze_dir),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--out-dir",
                    str(package_dir),
                ]
            )
            self.assertEqual(package.returncode, 2, msg=package.stdout + package.stderr)
            payload = json.loads(package.stdout)
            self.assertEqual(payload.get("error_type"), "output_dir_not_empty")


if __name__ == "__main__":
    unittest.main()
