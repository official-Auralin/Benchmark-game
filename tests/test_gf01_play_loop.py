"""
Regression tests for the playable GF-01 episode loop.

These tests validate that the `play` command runs in non-interactive mode for
baseline policies and emits a structured machine-checkable artifact.
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


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TestPlayableLoop(unittest.TestCase):
    def test_play_baseline_agent_emits_structured_payload(self) -> None:
        proc = _run_cli(["play", "--seed", "1337", "--agent", "greedy", "--renderer-track", "json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")
        run_contract = payload.get("run_contract", {})
        self.assertEqual(run_contract.get("renderer_track"), "json")
        self.assertEqual(run_contract.get("eval_track"), "EVAL-CB")
        self.assertEqual(run_contract.get("tool_allowlist_id"), "none")
        self.assertEqual(run_contract.get("play_protocol"), "commit_only")
        self.assertEqual(run_contract.get("scored_commit_episode"), True)
        self.assertEqual(run_contract.get("adaptation_condition"), "no_adaptation")
        self.assertEqual(run_contract.get("adaptation_budget_tokens"), 0)
        self.assertEqual(run_contract.get("adaptation_data_scope"), "none")
        self.assertEqual(run_contract.get("adaptation_protocol_id"), "none")
        episode = payload.get("episode", {})
        self.assertIn("certificate", episode)
        self.assertIn("suff", episode)
        self.assertIn("min1", episode)
        self.assertIn("valid", episode)
        self.assertIn("steps", episode)
        self.assertTrue(isinstance(episode.get("steps"), list))
        if episode["steps"]:
            first_step = episode["steps"][0]
            self.assertIn("action_set", first_step)
            self.assertIn("observation", first_step)
            self.assertIn("observation_rendered", first_step)

    def test_play_out_writes_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gf01-play-") as tmp:
            out = Path(tmp) / "episode.json"
            proc = _run_cli(
                [
                    "play",
                    "--seed",
                    "1337",
                    "--agent",
                    "greedy",
                    "--renderer-track",
                    "json",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "ok")
            self.assertIn("instance", payload)
            self.assertIn("episode", payload)

    def test_play_rejects_tool_agent_on_closed_book_track(self) -> None:
        proc = _run_cli(["play", "--seed", "1337", "--agent", "tool", "--eval-track", "EVAL-CB"])
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "track_policy_violation")

    def test_play_rejects_ta_track_without_tool_metadata(self) -> None:
        proc = _run_cli(["play", "--seed", "1337", "--agent", "tool", "--eval-track", "EVAL-TA"])
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "track_policy_violation")

    def test_play_allows_ta_track_with_tool_metadata(self) -> None:
        proc = _run_cli(
            [
                "play",
                "--seed",
                "1337",
                "--agent",
                "tool",
                "--eval-track",
                "EVAL-TA",
                "--tool-allowlist-id",
                "local-planner-v1",
                "--tool-log-hash",
                "demo-hash",
                "--renderer-track",
                "json",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")
        run_contract = payload.get("run_contract", {})
        self.assertEqual(run_contract.get("eval_track"), "EVAL-TA")
        self.assertEqual(run_contract.get("tool_allowlist_id"), "local-planner-v1")
        self.assertEqual(run_contract.get("play_protocol"), "commit_only")
        self.assertEqual(run_contract.get("scored_commit_episode"), True)
        self.assertEqual(run_contract.get("adaptation_condition"), "no_adaptation")

    def test_play_rejects_unknown_ta_allowlist(self) -> None:
        proc = _run_cli(
            [
                "play",
                "--seed",
                "1337",
                "--agent",
                "tool",
                "--eval-track",
                "EVAL-TA",
                "--tool-allowlist-id",
                "unknown-tools-v9",
                "--tool-log-hash",
                "demo-hash",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "track_policy_violation")

    def test_play_allows_oc_with_oracle_allowlist(self) -> None:
        proc = _run_cli(
            [
                "play",
                "--seed",
                "1337",
                "--agent",
                "oracle",
                "--eval-track",
                "EVAL-OC",
                "--tool-allowlist-id",
                "oracle-exact-search-v1",
                "--tool-log-hash",
                "oracle-demo-hash",
                "--renderer-track",
                "json",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("status"), "ok")
        run_contract = payload.get("run_contract", {})
        self.assertEqual(run_contract.get("eval_track"), "EVAL-OC")
        self.assertEqual(run_contract.get("tool_allowlist_id"), "oracle-exact-search-v1")

    def test_play_rejects_invalid_adaptation_combo(self) -> None:
        proc = _run_cli(
            [
                "play",
                "--seed",
                "1337",
                "--agent",
                "greedy",
                "--adaptation-condition",
                "weight_finetune",
                "--adaptation-budget-tokens",
                "0",
                "--adaptation-data-scope",
                "public_only",
                "--adaptation-protocol-id",
                "ft-run-v1",
            ]
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload.get("error_type"), "adaptation_policy_violation")


if __name__ == "__main__":
    unittest.main()
