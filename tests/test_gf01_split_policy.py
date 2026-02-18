"""
Regression tests for split-ratio governance policy checks.

These tests validate the split-policy checker used for publication governance
so split-manifest ratio drift is caught deterministically.
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
MANIFEST = FIXTURES / "split_manifest_v1.json"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestSplitPolicyCheck(unittest.TestCase):
    def test_split_policy_check_passes_with_matching_single_split_ratio(self) -> None:
        proc = _run_cli(
            [
                "split-policy-check",
                "--manifest",
                str(MANIFEST),
                "--target-ratios",
                "public_dev=1.0",
                "--tolerance",
                "0.0",
                "--private-split",
                "private_eval",
                "--min-private-eval-count",
                "0",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")

    def test_split_policy_check_fails_on_default_release_ratios_for_fixture(self) -> None:
        proc = _run_cli(
            [
                "split-policy-check",
                "--manifest",
                str(MANIFEST),
                "--target-ratios",
                "public_dev=0.2,public_val=0.2,private_eval=0.6",
                "--tolerance",
                "0.01",
                "--private-split",
                "private_eval",
                "--min-private-eval-count",
                "1",
                "--require-official-split-names",
                "--strict-manifest",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "split_policy_violation")
        self.assertFalse(payload.get("private_min_pass"))


if __name__ == "__main__":
    unittest.main()
