"""
Regression tests for adaptation/fine-tuning reporting policy enforcement.

These tests ensure adaptation condition labels and budget fields are validated
and machine-checkable in CLI execution paths.
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


class TestAdaptationPolicyCli(unittest.TestCase):
    def test_evaluate_rejects_nonzero_budget_for_no_adaptation(self) -> None:
        proc = _run_cli(
            [
                "evaluate",
                "--instances",
                str(BUNDLE),
                "--agent",
                "greedy",
                "--eval-track",
                "EVAL-CB",
                "--tool-allowlist-id",
                "none",
                "--adaptation-condition",
                "no_adaptation",
                "--adaptation-budget-tokens",
                "10",
                "--adaptation-data-scope",
                "none",
                "--adaptation-protocol-id",
                "none",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "adaptation_policy_violation")

    def test_evaluate_accepts_weight_finetune_with_budget_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-adapt-") as tmp:
            out = Path(tmp) / "adapt_runs.jsonl"
            proc = _run_cli(
                [
                    "evaluate",
                    "--instances",
                    str(BUNDLE),
                    "--agent",
                    "greedy",
                    "--eval-track",
                    "EVAL-CB",
                    "--tool-allowlist-id",
                    "none",
                    "--adaptation-condition",
                    "weight_finetune",
                    "--adaptation-budget-tokens",
                    "5000",
                    "--adaptation-data-scope",
                    "public_only",
                    "--adaptation-protocol-id",
                    "ft-public-v1",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows)
            self.assertTrue(all(r.get("adaptation_condition") == "weight_finetune" for r in rows))
            self.assertTrue(all(int(r.get("adaptation_budget_tokens", 0)) == 5000 for r in rows))
            self.assertTrue(all(r.get("adaptation_data_scope") == "public_only" for r in rows))
            self.assertTrue(all(r.get("adaptation_protocol_id") == "ft-public-v1" for r in rows))


if __name__ == "__main__":
    unittest.main()
