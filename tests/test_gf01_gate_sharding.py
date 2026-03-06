"""
Unit tests for GF-01 gate unittest sharding helpers.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gf01 import gate


class TestGateShardingHelpers(unittest.TestCase):
    def test_discover_unittest_modules_only_returns_test_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-gate-unit-") as tmp:
            root = Path(tmp)
            tests_dir = root / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            (tests_dir / "test_alpha.py").write_text("", encoding="utf-8")
            (tests_dir / "helper.py").write_text("", encoding="utf-8")
            (tests_dir / "test_beta.py").write_text("", encoding="utf-8")

            modules = gate._discover_unittest_modules(root)

            self.assertEqual(modules, ["tests.test_alpha", "tests.test_beta"])

    def test_partition_unittest_modules_assigns_every_module_once(self) -> None:
        modules = [
            "tests.test_gf01_release_candidate_check",
            "tests.test_gf01_release_package",
            "tests.test_gf01_pilot_analysis",
            "tests.test_small_alpha",
            "tests.test_small_beta",
        ]

        shards = gate._partition_unittest_modules(modules, 2)

        self.assertEqual(len(shards), 2)
        self.assertTrue(all(shard for shard in shards))
        self.assertCountEqual([module for shard in shards for module in shard], modules)
        for shard in shards:
            self.assertEqual(shard, sorted(shard))

    @patch("gf01.gate._discover_unittest_modules")
    @patch("gf01.gate._run_subprocess")
    def test_run_unittest_step_aggregates_shards(
        self,
        run_subprocess: unittest.mock.Mock,
        discover_unittest_modules: unittest.mock.Mock,
    ) -> None:
        discover_unittest_modules.return_value = [
            "tests.test_gf01_release_candidate_check",
            "tests.test_gf01_release_package",
            "tests.test_small_alpha",
        ]

        def fake_run(cmd: list[str], cwd: Path) -> dict[str, object]:
            self.assertEqual(cmd[:4], [sys.executable, "-m", "unittest", "-v"])
            return {
                "cmd": " ".join(cmd),
                "returncode": 0,
                "duration_ms": 12.5,
                "stdout_preview": [cmd[-1]],
                "stderr_preview": [],
                "stdout_text": "",
                "stderr_text": "",
            }

        run_subprocess.side_effect = fake_run

        result, notes = gate._run_unittest_step(Path("/tmp/project"), unittest_shards=2)

        self.assertEqual(run_subprocess.call_count, 2)
        self.assertEqual(result["cmd"], "sharded unittest x2")
        self.assertEqual(result["returncode"], 0)
        self.assertIsNotNone(notes)
        assert notes is not None
        self.assertEqual(notes["requested_shards"], 2)
        self.assertEqual(notes["effective_shards"], 2)
        self.assertEqual(notes["failed_shards"], [])
        self.assertEqual(len(notes["shards"]), 2)
        self.assertIn("[shard 1]", result["stdout_preview"][0])

    @patch("gf01.gate._discover_unittest_modules")
    @patch("gf01.gate._run_subprocess")
    def test_run_unittest_step_reports_failed_shards(
        self,
        run_subprocess: unittest.mock.Mock,
        discover_unittest_modules: unittest.mock.Mock,
    ) -> None:
        discover_unittest_modules.return_value = [
            "tests.test_alpha",
            "tests.test_beta",
        ]

        def fake_run(cmd: list[str], cwd: Path) -> dict[str, object]:
            failed = cmd[-1] == "tests.test_beta"
            return {
                "cmd": " ".join(cmd),
                "returncode": 1 if failed else 0,
                "duration_ms": 10.0,
                "stdout_preview": [],
                "stderr_preview": ["FAIL"] if failed else [],
                "stdout_text": "",
                "stderr_text": "FAIL" if failed else "",
            }

        run_subprocess.side_effect = fake_run

        result, notes = gate._run_unittest_step(Path("/tmp/project"), unittest_shards=2)

        self.assertEqual(result["returncode"], 1)
        self.assertIsNotNone(notes)
        assert notes is not None
        self.assertEqual(notes["failed_shards"], [2])
        self.assertEqual(len(notes["shards"]), 2)


if __name__ == "__main__":
    unittest.main()
