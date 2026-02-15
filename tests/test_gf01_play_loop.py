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
        self.assertEqual(payload.get("renderer_track"), "json")
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


if __name__ == "__main__":
    unittest.main()
