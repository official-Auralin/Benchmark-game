"""
Regression tests for CLI track/tool-allowlist policy enforcement.

These tests ensure `evaluate` enforces the same track policy contract used by
`play` and strict validation paths.
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
FIXTURES = ROOT / "tests" / "fixtures" / "official_example"
BUNDLE = FIXTURES / "instance_bundle_v1.json"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestTrackPolicyCli(unittest.TestCase):
    def test_evaluate_rejects_unknown_renderer_track(self) -> None:
        proc = _run_cli(
            [
                "evaluate",
                "--instances",
                str(BUNDLE),
                "--agent",
                "greedy",
                "--renderer-track",
                "barcode-v9",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "renderer_policy_violation")

    def test_evaluate_rejects_ta_without_tool_metadata(self) -> None:
        proc = _run_cli(
            [
                "evaluate",
                "--instances",
                str(BUNDLE),
                "--agent",
                "tool",
                "--eval-track",
                "EVAL-TA",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "track_policy_violation")

    def test_evaluate_rejects_unknown_ta_allowlist(self) -> None:
        proc = _run_cli(
            [
                "evaluate",
                "--instances",
                str(BUNDLE),
                "--agent",
                "tool",
                "--eval-track",
                "EVAL-TA",
                "--tool-allowlist-id",
                "unknown-tools-v9",
                "--tool-log-hash",
                "abc123",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "track_policy_violation")

    def test_evaluate_accepts_known_ta_allowlist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-track-policy-") as tmp:
            out = Path(tmp) / "runs.jsonl"
            proc = _run_cli(
                [
                    "evaluate",
                    "--instances",
                    str(BUNDLE),
                    "--agent",
                    "tool",
                    "--eval-track",
                    "EVAL-TA",
                    "--tool-allowlist-id",
                    "local-planner-v1",
                    "--tool-log-hash",
                    "abc123",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(out.exists())

    def test_evaluate_accepts_known_oc_allowlist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-track-policy-") as tmp:
            out = Path(tmp) / "runs_oracle.jsonl"
            proc = _run_cli(
                [
                    "evaluate",
                    "--instances",
                    str(BUNDLE),
                    "--agent",
                    "oracle",
                    "--eval-track",
                    "EVAL-OC",
                    "--tool-allowlist-id",
                    "oracle-exact-search-v1",
                    "--tool-log-hash",
                    "oracle-hash",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
