"""
Regression tests for the P0 seed-pack command wrapper.

This command should provide a deterministic shortcut over freeze-pilot for
internal-alpha setup.
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


class TestP0SeedPack(unittest.TestCase):
    def test_p0_seed_pack_writes_freeze_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-seed-pack-") as tmp:
            out_dir = Path(tmp) / "pack"
            proc = _run_cli(
                [
                    "p0-seed-pack",
                    "--seed-start",
                    "9100",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_dir),
                    "--force",
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("seed_count"), 2)
            self.assertEqual(payload.get("mode_override"), "normal")
            self.assertEqual(payload.get("split_id"), "pilot_internal_p0_v1")
            self.assertTrue((out_dir / "instance_bundle_v1.json").exists())
            self.assertTrue((out_dir / "split_manifest_v1.json").exists())
            self.assertTrue((out_dir / "pilot_freeze_v1.json").exists())

    def test_p0_seed_pack_rejects_nonempty_dir_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-seed-pack-") as tmp:
            out_dir = Path(tmp) / "pack"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "marker.txt").write_text("x\n", encoding="utf-8")
            proc = _run_cli(
                [
                    "p0-seed-pack",
                    "--seed-start",
                    "9100",
                    "--count",
                    "2",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "output_dir_not_empty")


if __name__ == "__main__":
    unittest.main()
