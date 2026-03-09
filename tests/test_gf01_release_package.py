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
__license__ = "Apache-2.0"
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


class TestReleasePackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._prepared_root = Path(
            tempfile.mkdtemp(prefix="gf01-release-package-fixtures-")
        )
        cls.addClassCleanup(shutil.rmtree, cls._prepared_root, ignore_errors=True)
        cls.freeze_dir = cls._prepared_root / "freeze"
        cls.campaign_dir = cls._prepared_root / "campaign"

        freeze = run_cli(
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
                str(cls.freeze_dir),
            ]
        )
        if freeze.returncode != 0:
            raise AssertionError(freeze.stdout + freeze.stderr)

        campaign = run_cli(
            [
                "pilot-campaign",
                "--freeze-dir",
                str(cls.freeze_dir),
                "--out-dir",
                str(cls.campaign_dir),
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
        if campaign.returncode != 0:
            raise AssertionError(campaign.stdout + campaign.stderr)

        analysis = run_cli(
            [
                "pilot-analyze",
                "--campaign-dir",
                str(cls.campaign_dir),
                "--eval-track",
                "EVAL-CB",
                "--mode",
                "normal",
            ]
        )
        if analysis.returncode != 0:
            raise AssertionError(analysis.stdout + analysis.stderr)

    def test_release_package_emits_manifest_and_instructions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-package-") as tmp:
            freeze_dir = clone_tree(self.freeze_dir, Path(tmp) / "freeze")
            campaign_dir = clone_tree(self.campaign_dir, Path(tmp) / "campaign")
            package_dir = Path(tmp) / "package"

            package = run_cli(
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
            freeze_dir = clone_tree(self.freeze_dir, Path(tmp) / "freeze")
            campaign_dir = clone_tree(self.campaign_dir, Path(tmp) / "campaign")
            package_dir = Path(tmp) / "package"
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "existing.txt").write_text("x", encoding="utf-8")

            package = run_cli(
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
