"""
Regression tests for the one-shot P0 initialization command.

This command should create the feedback template and deterministic seed pack in
one step so internal alpha setup is reproducible for non-expert operators.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

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


class TestP0Init(unittest.TestCase):
    def test_p0_init_writes_template_and_seed_pack(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-init-") as tmp:
            template_path = Path(tmp) / "p0_feedback.csv"
            out_dir = Path(tmp) / "pack"
            out_summary = Path(tmp) / "summary.json"
            proc = _run_cli(
                [
                    "p0-init",
                    "--template-out",
                    str(template_path),
                    "--seed-start",
                    "9200",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_dir),
                    "--out",
                    str(out_summary),
                    "--force",
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("passed"), True)
            self.assertTrue(template_path.exists())
            self.assertTrue((out_dir / "instance_bundle_v1.json").exists())
            self.assertTrue((out_dir / "split_manifest_v1.json").exists())
            self.assertTrue((out_dir / "pilot_freeze_v1.json").exists())
            self.assertTrue(out_summary.exists())
            written = json.loads(out_summary.read_text(encoding="utf-8"))
            self.assertEqual(written.get("schema_version"), "gf01.p0_init.v1")

    def test_p0_init_fails_when_template_exists_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-init-") as tmp:
            template_path = Path(tmp) / "p0_feedback.csv"
            template_path.write_text("existing\n", encoding="utf-8")
            out_dir = Path(tmp) / "pack"
            proc = _run_cli(
                [
                    "p0-init",
                    "--template-out",
                    str(template_path),
                    "--seed-start",
                    "9200",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_init_template_failed")
            self.assertFalse(out_dir.exists())

    def test_p0_init_fails_when_seed_pack_dir_nonempty_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-init-") as tmp:
            template_path = Path(tmp) / "p0_feedback.csv"
            out_dir = Path(tmp) / "pack"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "marker.txt").write_text("x\n", encoding="utf-8")
            proc = _run_cli(
                [
                    "p0-init",
                    "--template-out",
                    str(template_path),
                    "--seed-start",
                    "9200",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_init_seed_pack_failed")
            self.assertTrue(template_path.exists())


if __name__ == "__main__":
    unittest.main()
