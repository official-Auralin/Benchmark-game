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
    from .workflow_artifact_harness import (
        clone_tree,
        remap_freeze_to_official_splits,
        run_cli,
    )
except ImportError:  # pragma: no cover
    from workflow_artifact_harness import (
        clone_tree,
        remap_freeze_to_official_splits,
        run_cli,
    )


class TestReleaseCandidateCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._prepared_root = Path(
            tempfile.mkdtemp(prefix="gf01-release-candidate-fixtures-")
        )
        cls.addClassCleanup(shutil.rmtree, cls._prepared_root, ignore_errors=True)

        cls.current_freeze = cls._prepared_root / "current_freeze"
        cls.previous_freeze = cls._prepared_root / "previous_freeze"
        cls.full_campaign = cls._prepared_root / "campaign_full"
        cls.core_campaign = cls._prepared_root / "campaign_core"

        current = run_cli(
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
                str(cls.current_freeze),
            ]
        )
        if current.returncode != 0:
            raise AssertionError(current.stdout + current.stderr)

        previous = run_cli(
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
                str(cls.previous_freeze),
            ]
        )
        if previous.returncode != 0:
            raise AssertionError(previous.stdout + previous.stderr)

        remap_freeze_to_official_splits(cls.current_freeze)
        remap_freeze_to_official_splits(cls.previous_freeze)

        full_campaign = run_cli(
            [
                "pilot-campaign",
                "--freeze-dir",
                str(cls.current_freeze),
                "--out-dir",
                str(cls.full_campaign),
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
        if full_campaign.returncode != 0:
            raise AssertionError(full_campaign.stdout + full_campaign.stderr)

        core_campaign = run_cli(
            [
                "pilot-campaign",
                "--freeze-dir",
                str(cls.current_freeze),
                "--out-dir",
                str(cls.core_campaign),
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
        if core_campaign.returncode != 0:
            raise AssertionError(core_campaign.stdout + core_campaign.stderr)

    def test_release_candidate_check_passes_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-candidate-") as tmp:
            base = Path(tmp)
            current_freeze = clone_tree(self.current_freeze, base / "current_freeze")
            previous_freeze = clone_tree(self.previous_freeze, base / "previous_freeze")
            campaign_dir = clone_tree(self.full_campaign, base / "campaign")
            package_dir = base / "package"

            check = run_cli(
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
            current_freeze = clone_tree(self.current_freeze, base / "current_freeze")
            campaign_dir = clone_tree(self.core_campaign, base / "campaign")

            check = run_cli(
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
