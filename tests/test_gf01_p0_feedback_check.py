"""
Regression tests for the P0 feedback policy checker command.

These tests validate that internal-alpha feedback can be evaluated with a
machine-checkable pass/fail policy before advancing to formal participant runs.
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


class TestP0FeedbackCheck(unittest.TestCase):
    def _write_csv(self, path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_p0_feedback_check_passes_with_threshold_satisfied(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers",
                    "t01,4,4,4,0",
                    "t02,3,3,3,0",
                    "t03,5,4,3,0",
                    "t04,3,4,3,0",
                    "t05,4,3,4,0",
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("passed"), True)

    def test_p0_feedback_check_fails_on_must_fix_blockers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers",
                    "t01,4,4,4,1",
                    "t02,4,4,4,0",
                    "t03,4,4,4,0",
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "error")
            self.assertEqual(payload.get("passed"), False)
            self.assertEqual(payload.get("metrics", {}).get("must_fix_total"), 1)
            self.assertEqual(payload.get("error_type"), "feedback_thresholds_not_met")

    def test_p0_feedback_check_sets_error_type_when_ratios_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers",
                    "t01,2,2,2,0",
                    "t02,2,2,2,0",
                    "t03,2,2,2,0",
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "error")
            self.assertEqual(payload.get("passed"), False)
            self.assertEqual(payload.get("error_type"), "feedback_thresholds_not_met")

    def test_p0_feedback_check_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,must_fix_blockers",
                    "t01,4,4,0",
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "feedback_schema_error")

    def test_p0_feedback_check_rejects_non_integer_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers",
                    "t01,4,4,4,0",
                    "t02,three,4,4,0",
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "feedback_value_error")
            self.assertEqual(payload.get("bad_rows"), ["t02"])

    def test_p0_feedback_check_rejects_empty_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers"
                ],
            )
            proc = _run_cli(["p0-feedback-check", "--feedback", str(csv_path)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "empty_feedback")

    def test_p0_feedback_check_writes_out_json_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-feedback-") as tmp:
            csv_path = Path(tmp) / "feedback.csv"
            out_path = Path(tmp) / "feedback_check.json"
            self._write_csv(
                csv_path,
                [
                    "tester_id,objective_clarity,control_clarity,action_effect_clarity,must_fix_blockers",
                    "t01,4,4,4,0",
                    "t02,3,3,3,0",
                ],
            )
            proc = _run_cli(
                [
                    "p0-feedback-check",
                    "--feedback",
                    str(csv_path),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            stdout_payload = json.loads(proc.stdout)
            self.assertTrue(out_path.exists())
            written_payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written_payload, stdout_payload)
            self.assertEqual(written_payload.get("passed"), True)
            self.assertIn("metrics", written_payload)


if __name__ == "__main__":
    unittest.main()
