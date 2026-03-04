"""
Regression tests for the P0 feedback template command.

This command provides a deterministic CSV schema starter for internal alpha
playthrough collection.
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


class TestP0FeedbackTemplate(unittest.TestCase):
    def test_template_command_writes_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-template-") as tmp:
            out = Path(tmp) / "p0_feedback.csv"
            proc = _run_cli(["p0-feedback-template", "--out", str(out)])
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertTrue(out.exists())
            header = out.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("tester_id", header)
            self.assertIn("objective_clarity", header)
            self.assertIn("must_fix_blockers", header)

    def test_template_command_requires_force_for_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-template-") as tmp:
            out = Path(tmp) / "p0_feedback.csv"
            out.write_text("x\n", encoding="utf-8")
            proc = _run_cli(["p0-feedback-template", "--out", str(out)])
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "template_exists")


if __name__ == "__main__":
    unittest.main()

