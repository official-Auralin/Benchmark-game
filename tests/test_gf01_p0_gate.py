"""
Regression tests for the combined P0 gate command.

The gate should enforce session-artifact coverage before evaluating feedback
thresholds and emit one machine-checkable pass/fail summary.
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


def _write_feedback_csv(path: Path, rows: list[str]) -> None:
    header = (
        "tester_id,date,backend_used,seed_list_run,objective_clarity,"
        "control_clarity,action_effect_clarity,visual_overload,must_fix_blockers,notes"
    )
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _write_play_payload(path: Path, seed: int, backend: str = "pygame") -> None:
    payload = {
        "status": "ok",
        "instance": {"seed": seed},
        "run_contract": {
            "renderer_track": "visual",
            "visual_backend": backend,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TestP0Gate(unittest.TestCase):
    def test_p0_gate_passes_when_session_and_feedback_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-gate-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                [
                    "tester_01,2026-03-03,pygame,7000;7001,3,3,3,2,0,",
                    "tester_02,2026-03-03,text,7000,4,4,4,2,0,",
                ],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")
            _write_play_payload(runs_dir / "tester_01_7001.json", 7001, backend="pygame")
            _write_play_payload(runs_dir / "tester_02_7000.json", 7000, backend="text")

            proc = _run_cli(
                [
                    "p0-gate",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("passed"), True)
            self.assertEqual(payload.get("schema_version"), "gf01.p0_gate.v1")

    def test_p0_gate_fails_when_session_check_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-gate-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000;7001,3,3,3,2,0,"],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")

            proc = _run_cli(
                [
                    "p0-gate",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_gate_session_check_failed")
            self.assertEqual(payload.get("passed"), False)

    def test_p0_gate_fails_when_feedback_check_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-gate-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,4,4,4,2,1,"],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")

            proc = _run_cli(
                [
                    "p0-gate",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_gate_feedback_check_failed")
            self.assertEqual(payload.get("passed"), False)

    def test_p0_gate_writes_out_summary_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-gate-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            out_path = root / "p0_gate_summary.json"
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")

            proc = _run_cli(
                [
                    "p0-gate",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(out_path.exists())
            summary = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(summary.get("schema_version"), "gf01.p0_gate.v1")
            self.assertEqual(summary.get("passed"), True)


if __name__ == "__main__":
    unittest.main()
