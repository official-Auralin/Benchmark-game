"""
Regression tests for release-governance protocol checks.

These tests validate split-policy and rotation-contamination checks that are
used to gate publication-cycle release manifests.
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


def _manifest_payload(entries: list[tuple[str, str]]) -> dict[str, object]:
    rows = []
    for instance_id, split_id in entries:
        rows.append(
            {
                "instance_id": instance_id,
                "split_id": split_id,
                "mode": "normal",
                "seed": 0,
                "t_star": 1,
                "window_size": 1,
                "budget_timestep": 1,
                "budget_atoms": 1,
            }
        )
    return {
        "schema_version": "gf01.split_manifest.v1",
        "family_id": "GF-01",
        "benchmark_version": "0.1.0-dev",
        "generator_version": "0.1.0-dev",
        "checker_version": "0.1.0-dev",
        "harness_version": "1.0.0",
        "instance_count": len(rows),
        "instances": rows,
    }


class TestReleaseGovernanceCheck(unittest.TestCase):
    def test_release_governance_passes_with_rotation_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-governance-") as tmp:
            current_path = Path(tmp) / "current_manifest.json"
            previous_path = Path(tmp) / "previous_manifest.json"

            current_entries = [
                ("cur_pub_dev_0", "public_dev"),
                ("cur_pub_dev_1", "public_dev"),
                ("cur_pub_val_0", "public_val"),
                ("cur_pub_val_1", "public_val"),
            ] + [(f"cur_priv_{i}", "private_eval") for i in range(6)]
            prev_entries = [
                ("prev_pub_dev_0", "public_dev"),
                ("prev_pub_dev_1", "public_dev"),
                ("prev_pub_val_0", "public_val"),
                ("prev_pub_val_1", "public_val"),
            ] + [(f"prev_priv_{i}", "private_eval") for i in range(6)]

            current_path.write_text(
                json.dumps(_manifest_payload(current_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            previous_path.write_text(
                json.dumps(_manifest_payload(prev_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            proc = _run_cli(
                [
                    "release-governance-check",
                    "--manifest",
                    str(current_path),
                    "--previous-manifest",
                    str(previous_path),
                    "--require-previous-manifest",
                    "--min-public-novelty-ratio",
                    "0.50",
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            rotation = payload.get("rotation_policy", {})
            self.assertTrue(rotation.get("public_novelty_pass"))
            self.assertTrue(rotation.get("private_to_public_pass"))

    def test_release_governance_fails_on_private_to_public_leakage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-governance-") as tmp:
            current_path = Path(tmp) / "current_manifest.json"
            previous_path = Path(tmp) / "previous_manifest.json"

            current_entries = [
                ("prev_priv_0", "public_dev"),
                ("cur_pub_dev_1", "public_dev"),
                ("cur_pub_val_0", "public_val"),
                ("cur_pub_val_1", "public_val"),
            ] + [(f"cur_priv_{i}", "private_eval") for i in range(6)]
            prev_entries = [
                ("prev_pub_dev_0", "public_dev"),
                ("prev_pub_dev_1", "public_dev"),
                ("prev_pub_val_0", "public_val"),
                ("prev_pub_val_1", "public_val"),
            ] + [(f"prev_priv_{i}", "private_eval") for i in range(6)]

            current_path.write_text(
                json.dumps(_manifest_payload(current_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            previous_path.write_text(
                json.dumps(_manifest_payload(prev_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            proc = _run_cli(
                [
                    "release-governance-check",
                    "--manifest",
                    str(current_path),
                    "--previous-manifest",
                    str(previous_path),
                    "--require-previous-manifest",
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "release_governance_violation")
            rotation = payload.get("rotation_policy", {})
            self.assertFalse(rotation.get("private_to_public_pass"))

    def test_release_governance_fails_on_low_public_novelty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-governance-") as tmp:
            current_path = Path(tmp) / "current_manifest.json"
            previous_path = Path(tmp) / "previous_manifest.json"

            current_entries = [
                ("shared_pub_dev_0", "public_dev"),
                ("shared_pub_dev_1", "public_dev"),
                ("shared_pub_val_0", "public_val"),
                ("shared_pub_val_1", "public_val"),
            ] + [(f"cur_priv_{i}", "private_eval") for i in range(6)]
            prev_entries = [
                ("shared_pub_dev_0", "public_dev"),
                ("shared_pub_dev_1", "public_dev"),
                ("shared_pub_val_0", "public_val"),
                ("shared_pub_val_1", "public_val"),
            ] + [(f"prev_priv_{i}", "private_eval") for i in range(6)]

            current_path.write_text(
                json.dumps(_manifest_payload(current_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            previous_path.write_text(
                json.dumps(_manifest_payload(prev_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            proc = _run_cli(
                [
                    "release-governance-check",
                    "--manifest",
                    str(current_path),
                    "--previous-manifest",
                    str(previous_path),
                    "--require-previous-manifest",
                    "--min-public-novelty-ratio",
                    "0.25",
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "release_governance_violation")
            rotation = payload.get("rotation_policy", {})
            self.assertFalse(rotation.get("public_novelty_pass"))

    def test_release_governance_enforces_previous_manifest_when_required(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-release-governance-") as tmp:
            current_path = Path(tmp) / "current_manifest.json"
            current_entries = [
                ("cur_pub_dev_0", "public_dev"),
                ("cur_pub_dev_1", "public_dev"),
                ("cur_pub_val_0", "public_val"),
                ("cur_pub_val_1", "public_val"),
            ] + [(f"cur_priv_{i}", "private_eval") for i in range(6)]
            current_path.write_text(
                json.dumps(_manifest_payload(current_entries), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            proc = _run_cli(
                [
                    "release-governance-check",
                    "--manifest",
                    str(current_path),
                    "--require-previous-manifest",
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(
                payload.get("error_type"),
                "release_rotation_previous_manifest_required",
            )


if __name__ == "__main__":
    unittest.main()
