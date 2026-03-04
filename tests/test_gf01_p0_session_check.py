"""
Regression tests for the P0 session coverage checker.

The checker ensures feedback declarations map to concrete play artifacts with
matching seed/backend metadata before advancing the internal alpha gate.
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


def _write_play_payload(
    path: Path, seed: int, backend: str = "pygame", renderer_track: str = "visual"
) -> None:
    payload = {
        "status": "ok",
        "instance": {"seed": seed},
        "run_contract": {
            "renderer_track": renderer_track,
            "visual_backend": backend,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TestP0SessionCheck(unittest.TestCase):
    def test_session_check_passes_with_complete_matching_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                [
                    "tester_01,2026-03-03,pygame,7000;7001,3,3,3,2,0,",
                    "tester_02,2026-03-03,text,7000,3,3,3,2,0,",
                ],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")
            _write_play_payload(runs_dir / "tester_01_7001.json", 7001, backend="pygame")
            _write_play_payload(runs_dir / "tester_02_7000.json", 7000, backend="text")

            proc = _run_cli(
                [
                    "p0-session-check",
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
            self.assertEqual(payload.get("missing_session_count"), 0)
            self.assertEqual(payload.get("payload_issue_count"), 0)

    def test_session_check_fails_when_session_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
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
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_session_coverage_failed")
            self.assertEqual(payload.get("missing_session_count"), 1)

    def test_session_check_fails_on_backend_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="text")

            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_session_coverage_failed")
            self.assertEqual(payload.get("payload_issue_count"), 1)

    def test_session_check_accepts_mixed_seed_separators(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ['tester_01,2026-03-03,pygame,"7000, 7001 7002;7003",3,3,3,2,0,'],
            )
            for seed in (7000, 7001, 7002, 7003):
                _write_play_payload(
                    runs_dir / f"tester_01_{seed}.json", seed, backend="pygame"
                )

            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("expected_session_count"), 4)

    def test_session_check_rejects_invalid_seed_list_format(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000;abc,3,3,3,2,0,"],
            )
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "feedback_seed_list_error")
            self.assertEqual(payload.get("bad_rows"), ["tester_01"])

    def test_session_check_rejects_renderer_track_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            _write_play_payload(
                runs_dir / "tester_01_7000.json",
                7000,
                backend="pygame",
                renderer_track="json",
            )
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_session_coverage_failed")
            issues = payload.get("payload_issues_preview", [])
            self.assertTrue(any(i.get("issue") == "renderer_track_mismatch" for i in issues))

    def test_session_check_rejects_invalid_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            (runs_dir / "tester_01_7000.json").write_text("{bad-json", encoding="utf-8")
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_session_coverage_failed")
            issues = payload.get("payload_issues_preview", [])
            self.assertTrue(any(i.get("issue") == "invalid_json" for i in issues))

    def test_session_check_rejects_non_object_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            (runs_dir / "tester_01_7000.json").write_text(
                json.dumps([], indent=2), encoding="utf-8"
            )
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "p0_session_coverage_failed")
            issues = payload.get("payload_issues_preview", [])
            self.assertTrue(any(i.get("issue") == "payload_not_object" for i in issues))

    def test_session_check_rejects_missing_feedback_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "missing-feedback.csv"
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            _write_play_payload(runs_dir / "tester_01_7000.json", 7000, backend="pygame")
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "missing_feedback_file")

    def test_session_check_rejects_missing_runs_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-p0-session-") as tmp:
            root = Path(tmp)
            csv_path = root / "feedback.csv"
            runs_dir = root / "missing-runs"
            _write_feedback_csv(
                csv_path,
                ["tester_01,2026-03-03,pygame,7000,3,3,3,2,0,"],
            )
            proc = _run_cli(
                [
                    "p0-session-check",
                    "--feedback",
                    str(csv_path),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "missing_runs_dir")


if __name__ == "__main__":
    unittest.main()
