"""
Regression tests for Q-033 manifest/sweep/closure command workflow.

These tests keep sample sizes small to verify command wiring and schema-level
behavior deterministically without attempting full high-performance closure.
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


class TestQ033ProtocolWorkflow(unittest.TestCase):
    def test_q033_manifest_sweep_and_closure_smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-q033-") as tmp:
            out_dir = Path(tmp) / "manifests"
            build = _run_cli(
                [
                    "q033-build-manifests",
                    "--seed-start",
                    "9500",
                    "--candidate-count",
                    "80",
                    "--replicates",
                    "2",
                    "--per-quartile",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(build.returncode, 0, msg=build.stdout + build.stderr)
            build_payload = json.loads(build.stdout)
            self.assertEqual(build_payload.get("status"), "ok")
            manifests = build_payload.get("manifest_paths", [])
            self.assertEqual(len(manifests), 2)

            sweep_paths: list[str] = []
            for idx, manifest_path in enumerate(manifests, start=1):
                sweep_out = Path(tmp) / f"sweep_{idx}.json"
                sweep = _run_cli(
                    [
                        "q033-sweep",
                        "--manifest",
                        str(manifest_path),
                        "--seed",
                        str(1600 + idx),
                        "--max-generate-ms-mean",
                        "100000",
                        "--max-minset-ms-mean",
                        "100000",
                        "--max-eval-ms-mean",
                        "100000",
                        "--max-checks-total-ms",
                        "100000",
                        "--max-truncation-rate",
                        "1.0",
                        "--min-oracle-minus-greedy-gap",
                        "-1.0",
                        "--max-quartile-truncation-rate",
                        "1.0",
                        "--min-quartile-gap",
                        "-1.0",
                        "--max-quartile-runtime-gate-failures",
                        "4",
                        "--out",
                        str(sweep_out),
                    ]
                )
                self.assertEqual(sweep.returncode, 0, msg=sweep.stdout + sweep.stderr)
                sweep_payload = json.loads(sweep.stdout)
                self.assertEqual(sweep_payload.get("status"), "ok")
                self.assertTrue(sweep_payload.get("gates", {}).get("passed_all", False))
                sweep_paths.append(str(sweep_out))

            close = _run_cli(
                [
                    "q033-closure-check",
                    "--sweep",
                    sweep_paths[0],
                    "--sweep",
                    sweep_paths[1],
                ]
            )
            self.assertEqual(close.returncode, 0, msg=close.stdout + close.stderr)
            close_payload = json.loads(close.stdout)
            self.assertTrue(close_payload.get("close_q033"))
            self.assertTrue(close_payload.get("disjoint_ok"))

    def test_q033_closure_detects_seed_overlap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-q033-") as tmp:
            out_dir = Path(tmp) / "manifests"
            build = _run_cli(
                [
                    "q033-build-manifests",
                    "--seed-start",
                    "9600",
                    "--candidate-count",
                    "80",
                    "--replicates",
                    "2",
                    "--per-quartile",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(build.returncode, 0, msg=build.stdout + build.stderr)
            manifest_path = json.loads(build.stdout).get("manifest_paths", [])[0]
            self.assertTrue(manifest_path)

            sweep_out = Path(tmp) / "sweep_same.json"
            sweep = _run_cli(
                [
                    "q033-sweep",
                    "--manifest",
                    str(manifest_path),
                    "--seed",
                    "1700",
                    "--max-generate-ms-mean",
                    "100000",
                    "--max-minset-ms-mean",
                    "100000",
                    "--max-eval-ms-mean",
                    "100000",
                    "--max-checks-total-ms",
                    "100000",
                    "--max-truncation-rate",
                    "1.0",
                    "--min-oracle-minus-greedy-gap",
                    "-1.0",
                    "--max-quartile-truncation-rate",
                    "1.0",
                    "--min-quartile-gap",
                    "-1.0",
                    "--max-quartile-runtime-gate-failures",
                    "4",
                    "--out",
                    str(sweep_out),
                ]
            )
            self.assertEqual(sweep.returncode, 0, msg=sweep.stdout + sweep.stderr)

            close = _run_cli(
                [
                    "q033-closure-check",
                    "--sweep",
                    str(sweep_out),
                    "--sweep",
                    str(sweep_out),
                ]
            )
            self.assertEqual(close.returncode, 2, msg=close.stdout + close.stderr)
            close_payload = json.loads(close.stdout)
            self.assertFalse(close_payload.get("close_q033"))
            self.assertFalse(close_payload.get("disjoint_ok"))


if __name__ == "__main__":
    unittest.main()
