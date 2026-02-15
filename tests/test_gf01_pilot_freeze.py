"""
Regression tests for provisional pilot-pack freeze workflow.

These tests verify that `freeze-pilot` produces deterministic, structured
artifacts for internal pilot reproducibility and respects overwrite guards.
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


class TestPilotFreeze(unittest.TestCase):
    def test_freeze_pilot_writes_bundle_manifest_and_meta(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-freeze-") as tmp:
            out_dir = Path(tmp) / "pilot"
            proc = _run_cli(
                [
                    "freeze-pilot",
                    "--freeze-id",
                    "gf01-pilot-freeze-test",
                    "--split",
                    "pilot_internal_test",
                    "--seed-start",
                    "9100",
                    "--count",
                    "3",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            summary = json.loads(proc.stdout)
            self.assertEqual(summary.get("status"), "ok")
            self.assertEqual(summary.get("instance_count"), 3)

            bundle_path = Path(summary["bundle_path"])
            manifest_path = Path(summary["manifest_path"])
            freeze_path = Path(summary["freeze_meta_path"])
            self.assertTrue(bundle_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(freeze_path.exists())

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            freeze_meta = json.loads(freeze_path.read_text(encoding="utf-8"))

            self.assertEqual(bundle.get("schema_version"), "gf01.instance_bundle.v1")
            self.assertEqual(len(bundle.get("instances", [])), 3)
            self.assertEqual(manifest.get("schema_version"), "gf01.split_manifest.v1")
            self.assertEqual(manifest.get("instance_count"), 3)
            self.assertEqual(freeze_meta.get("schema_version"), "gf01.pilot_freeze.v1")
            self.assertEqual(freeze_meta.get("freeze_id"), "gf01-pilot-freeze-test")
            self.assertEqual(freeze_meta.get("seed_count"), 3)
            self.assertTrue(freeze_meta.get("provisional"))

    def test_freeze_pilot_requires_force_for_non_empty_output_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-freeze-") as tmp:
            out_dir = Path(tmp) / "pilot"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "preexisting.txt").write_text("x", encoding="utf-8")
            proc = _run_cli(
                [
                    "freeze-pilot",
                    "--out-dir",
                    str(out_dir),
                    "--seed-start",
                    "9100",
                    "--count",
                    "2",
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "output_dir_not_empty")


if __name__ == "__main__":
    unittest.main()
