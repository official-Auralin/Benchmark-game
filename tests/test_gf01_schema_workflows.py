"""
Regression tests for GF-01 validation and legacy migration workflows.

These tests use pinned fixture artifacts so command behavior is stable across
runs. They exercise the CLI-level schema checks added in G9/G10 and confirm
legacy backfill enables strict/official validation paths.
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
MANIFEST = FIXTURES / "split_manifest_v1.json"
RUNS_V1 = FIXTURES / "runs_v1_valid.jsonl"
RUNS_LEGACY = FIXTURES / "runs_pre_v1_legacy.jsonl"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestSchemaWorkflows(unittest.TestCase):
    def test_validate_official_passes_on_v1_fixture(self) -> None:
        proc = _run_cli(
            [
                "validate",
                "--runs",
                str(RUNS_V1),
                "--manifest",
                str(MANIFEST),
                "--official",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")
        self.assertTrue(payload.get("strict_mode"))
        self.assertEqual(payload.get("rows"), 4)

    def test_validate_official_requires_manifest(self) -> None:
        proc = _run_cli(["validate", "--runs", str(RUNS_V1), "--official"])
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "official_manifest_required")

    def test_validate_strict_fails_on_legacy_fixture(self) -> None:
        proc = _run_cli(
            [
                "validate",
                "--runs",
                str(RUNS_LEGACY),
                "--manifest",
                str(MANIFEST),
                "--strict",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "run_schema_validation")
        self.assertGreater(int(payload.get("error_count", 0)), 0)

    def test_migrate_legacy_enables_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-test-") as tmp:
            migrated = Path(tmp) / "migrated_runs.jsonl"
            mig = _run_cli(
                [
                    "migrate-runs",
                    "--runs",
                    str(RUNS_LEGACY),
                    "--out",
                    str(migrated),
                    "--manifest",
                    str(MANIFEST),
                ]
            )
            self.assertEqual(mig.returncode, 0, msg=mig.stdout + mig.stderr)
            mig_payload = json.loads(mig.stdout)
            self.assertEqual(mig_payload.get("status"), "ok")
            self.assertEqual(mig_payload.get("migration_stats", {}).get("input_rows"), 4)
            self.assertEqual(mig_payload.get("migration_stats", {}).get("output_rows"), 4)

            val = _run_cli(
                [
                    "validate",
                    "--runs",
                    str(migrated),
                    "--manifest",
                    str(MANIFEST),
                    "--official",
                ]
            )
            self.assertEqual(val.returncode, 0, msg=val.stdout + val.stderr)
            val_payload = json.loads(val.stdout)
            self.assertEqual(val_payload.get("status"), "ok")
            self.assertEqual(val_payload.get("rows"), 4)

            schema_values = []
            for line in migrated.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    schema_values.append(json.loads(line).get("schema_version"))
            self.assertTrue(schema_values)
            self.assertTrue(all(v == "gf01.run_record.v1" for v in schema_values))

    def test_report_official_passes_on_v1_fixture(self) -> None:
        proc = _run_cli(
            [
                "report",
                "--runs",
                str(RUNS_V1),
                "--manifest",
                str(MANIFEST),
                "--official",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("strict_mode"), True)
        self.assertEqual(payload.get("total_rows"), 4)
        groups = payload.get("groups", [])
        self.assertTrue(groups)
        first = groups[0]
        self.assertIn("play_protocol", first)
        self.assertIn("scored_commit_episode", first)

    def test_validate_strict_fails_on_bad_ta_tool_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-test-") as tmp:
            bad_runs = Path(tmp) / "bad_ta_runs.jsonl"
            rows = []
            for line in RUNS_V1.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            self.assertTrue(rows)
            rows[0]["eval_track"] = "EVAL-TA"
            rows[0]["tool_allowlist_id"] = "none"
            rows[0]["tool_log_hash"] = ""
            bad_runs.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            proc = _run_cli(
                [
                    "validate",
                    "--runs",
                    str(bad_runs),
                    "--manifest",
                    str(MANIFEST),
                    "--strict",
                ]
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("error_type"), "run_schema_validation")


if __name__ == "__main__":
    unittest.main()
