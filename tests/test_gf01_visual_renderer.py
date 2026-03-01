"""
Regression tests for GF-01 visual renderer readability and parse parity.

These tests ensure the human-facing visual rendering remains machine-parsable
to the same canonical observation object and preserves backward compatibility
for legacy visual strings.
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

import unittest

from gf01.semantics import parse_visual, render_json, render_visual


class TestVisualRenderer(unittest.TestCase):
    def test_visual_renderer_includes_human_summary_and_parse_anchor(self) -> None:
        obs = {
            "t": 3,
            "y_t": {"out0": 1, "out1": 0},
            "effect_status_t": "not-triggered",
            "budget_t_remaining": 2,
            "budget_a_remaining": 4,
            "history_atoms": [(0, "in0", 1), (2, "in1", 0)],
            "mode": "normal",
            "t_star": 5,
        }
        rendered = render_visual(obs)
        self.assertIn("=== GF-01 Visual Snapshot ===", rendered)
        self.assertIn("Time: t=3 (target t*=5, mode=normal)", rendered)
        self.assertIn("Budget remaining: timesteps=2, atoms=4", rendered)
        self.assertIn("Outputs y_t: out0=1 out1=0", rendered)
        self.assertIn("Interventions so far:", rendered)
        self.assertIn("OBS_JSON=", rendered)

    def test_visual_roundtrip_matches_canonical_json(self) -> None:
        obs = {
            "t": 1,
            "y_t": {"out0": 0},
            "effect_status_t": "triggered",
            "budget_t_remaining": 1,
            "budget_a_remaining": 3,
            "history_atoms": [(0, "in0", 1)],
            "mode": "hard",
            "t_star": 1,
        }
        parsed = parse_visual(render_visual(obs))
        self.assertEqual(render_json(parsed), render_json(obs))

    def test_parse_visual_accepts_legacy_key_value_format(self) -> None:
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={\"out0\":1}",
                "BT=1",
                "BA=2",
                "H=[[0,\"in0\",1]]",
            ]
        )
        parsed = parse_visual(rendered)
        expected = {
            "t": 2,
            "mode": "normal",
            "t_star": 4,
            "effect_status_t": "not-triggered",
            "y_t": {"out0": 1},
            "budget_t_remaining": 1,
            "budget_a_remaining": 2,
            "history_atoms": [[0, "in0", 1]],
        }
        self.assertEqual(parsed, expected)


if __name__ == "__main__":
    unittest.main()
