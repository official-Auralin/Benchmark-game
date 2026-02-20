"""
Regression tests for identifiability-policy threshold enforcement.

These tests validate the machine-checkable policy gate used to ensure generated
or frozen instances remain informative under partial observability. The command
must pass at default policy thresholds on the official fixture and fail when
thresholds are intentionally made impossible.
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


class TestIdentifiabilityPolicy(unittest.TestCase):
    def test_identifiability_check_passes_on_fixture_defaults(self) -> None:
        proc = _run_cli(
            [
                "identifiability-check",
                "--instances",
                str(BUNDLE),
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(int(payload.get("fail_count", 1)), 0)
        self.assertEqual(int(payload.get("instance_count", 0)), 4)

    def test_identifiability_check_fails_on_impossible_thresholds(self) -> None:
        proc = _run_cli(
            [
                "identifiability-check",
                "--instances",
                str(BUNDLE),
                "--min-response-ratio",
                "1.0",
                "--min-unique-signatures",
                "999",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "identifiability_policy_violation")
        self.assertGreater(int(payload.get("fail_count", 0)), 0)


if __name__ == "__main__":
    unittest.main()
