"""
Regression tests for byte-stable canonical artifact workflows.

These checks ensure official/publication-facing commands emit canonical files
that are identical across repeated runs with the same inputs and repo state.
Non-canonical operator receipts are asserted separately and are excluded from
byte-for-byte comparisons.
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
import tempfile
import unittest
from pathlib import Path

try:
    from .workflow_artifact_harness import q033_manifest_paths, run_cli
except ImportError:  # pragma: no cover
    from workflow_artifact_harness import q033_manifest_paths, run_cli


def _assert_same_bytes(
    test_case: unittest.TestCase,
    left_root: Path,
    right_root: Path,
    *,
    names: list[str],
) -> None:
    for name in names:
        left = left_root / name
        right = right_root / name
        test_case.assertTrue(left.exists(), msg=f"missing canonical artifact: {left}")
        test_case.assertTrue(right.exists(), msg=f"missing canonical artifact: {right}")
        test_case.assertEqual(
            left.read_bytes(),
            right.read_bytes(),
            msg=f"canonical artifact bytes differ for {name}",
        )


class TestArtifactDeterminism(unittest.TestCase):
    def test_generate_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-generate-det-") as tmp:
            tmp_path = Path(tmp)
            bundle_a = tmp_path / "bundle_a.json"
            bundle_b = tmp_path / "bundle_b.json"
            manifest_a = tmp_path / "manifest_a.json"
            manifest_b = tmp_path / "manifest_b.json"

            first = run_cli(
                [
                    "generate",
                    "--seed",
                    "7100",
                    "--count",
                    "4",
                    "--split",
                    "public_dev",
                    "--out",
                    str(bundle_a),
                    "--manifest-out",
                    str(manifest_a),
                ]
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)

            second = run_cli(
                [
                    "generate",
                    "--seed",
                    "7100",
                    "--count",
                    "4",
                    "--split",
                    "public_dev",
                    "--out",
                    str(bundle_b),
                    "--manifest-out",
                    str(manifest_b),
                ]
            )
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)

            self.assertEqual(bundle_a.read_bytes(), bundle_b.read_bytes())
            self.assertEqual(manifest_a.read_bytes(), manifest_b.read_bytes())

    def test_freeze_pilot_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-freeze-det-") as tmp:
            tmp_path = Path(tmp)
            out_a = tmp_path / "freeze_a"
            out_b = tmp_path / "freeze_b"

            first = run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-determinism-freeze",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_a),
                ]
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            second = run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-determinism-freeze",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9300",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_b),
                ]
            )
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)

            _assert_same_bytes(
                self,
                out_a,
                out_b,
                names=[
                    "instance_bundle_v1.json",
                    "split_manifest_v1.json",
                    "pilot_freeze_v1.json",
                ],
            )
            self.assertTrue((out_a / "build_receipt.json").exists())
            self.assertTrue((out_b / "build_receipt.json").exists())

    def test_pilot_campaign_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-campaign-det-") as tmp:
            tmp_path = Path(tmp)
            freeze_dir = tmp_path / "freeze"
            freeze = run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-determinism-campaign",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9310",
                    "--count",
                    "2",
                    "--out-dir",
                    str(freeze_dir),
                ]
            )
            self.assertEqual(freeze.returncode, 0, msg=freeze.stdout + freeze.stderr)

            out_a = tmp_path / "campaign_a"
            out_b = tmp_path / "campaign_b"
            command = [
                "pilot-campaign",
                "--freeze-dir",
                str(freeze_dir),
                "--baseline-panel",
                "random,greedy,oracle",
                "--baseline-policy-level",
                "core",
                "--renderer-track",
                "json",
                "--seed",
                "55",
            ]
            first = run_cli([*command, "--out-dir", str(out_a)])
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            second = run_cli([*command, "--out-dir", str(out_b)])
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)

            _assert_same_bytes(
                self,
                out_a,
                out_b,
                names=[
                    "runs_combined.jsonl",
                    "official_validation.json",
                    "official_report.json",
                ],
            )
            self.assertTrue((out_a / "build_receipt.json").exists())
            self.assertTrue((out_b / "build_receipt.json").exists())

    def test_release_package_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-package-det-") as tmp:
            tmp_path = Path(tmp)
            freeze_dir = tmp_path / "freeze"
            campaign_dir = tmp_path / "campaign"

            freeze = run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-determinism-package",
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

            campaign = run_cli(
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

            analysis = run_cli(
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

            out_a = tmp_path / "package_a"
            out_b = tmp_path / "package_b"
            first = run_cli(
                [
                    "release-package",
                    "--freeze-dir",
                    str(freeze_dir),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--out-dir",
                    str(out_a),
                ]
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            second = run_cli(
                [
                    "release-package",
                    "--freeze-dir",
                    str(freeze_dir),
                    "--campaign-dir",
                    str(campaign_dir),
                    "--out-dir",
                    str(out_b),
                ]
            )
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)

            _assert_same_bytes(
                self,
                out_a,
                out_b,
                names=[
                    "release_package_manifest.json",
                    "RERUN_INSTRUCTIONS.md",
                ],
            )
            _assert_same_bytes(
                self,
                out_a / "artifacts",
                out_b / "artifacts",
                names=[
                    "instance_bundle_v1.json",
                    "split_manifest_v1.json",
                    "pilot_freeze_v1.json",
                    "runs_combined.jsonl",
                    "official_validation.json",
                    "official_report.json",
                    "pilot_analysis.json",
                ],
            )
            self.assertTrue((out_a / "build_receipt.json").exists())
            self.assertTrue((out_b / "build_receipt.json").exists())

    def test_q033_build_manifests_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-q033-det-") as tmp:
            tmp_path = Path(tmp)
            out_a = tmp_path / "q033_a"
            out_b = tmp_path / "q033_b"
            command = [
                "q033-build-manifests",
                "--seed-start",
                "9500",
                "--candidate-count",
                "80",
                "--replicates",
                "2",
                "--per-quartile",
                "1",
            ]

            first = run_cli([*command, "--out-dir", str(out_a)])
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            second = run_cli([*command, "--out-dir", str(out_b)])
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)

            index_a = json.loads((out_a / "q033_manifest_index.json").read_text(encoding="utf-8"))
            index_b = json.loads((out_b / "q033_manifest_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index_a, index_b)
            self.assertTrue((out_a / "build_receipt.json").exists())
            self.assertTrue((out_b / "build_receipt.json").exists())

            for path_a, path_b in zip(q033_manifest_paths(out_a), q033_manifest_paths(out_b)):
                self.assertEqual(path_a.name, path_b.name)
                self.assertEqual(path_a.read_bytes(), path_b.read_bytes(), msg=path_a.name)


if __name__ == "__main__":
    unittest.main()
