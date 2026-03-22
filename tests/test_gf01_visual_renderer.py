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
__license__ = "Apache-2.0"
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
            "y_t": {"out1": 0, "out0": 1},
            "effect_status_t": "not-triggered",
            "budget_t_remaining": 2,
            "budget_a_remaining": 4,
            "history_atoms": [
                (2, "in1", 0),
                (0, "in1", 0),
                (0, "in0", 1),
                (2, "in0", 1),
            ],
            "mode": "normal",
            "t_star": 5,
        }
        rendered = render_visual(obs)
        self.assertIn("=== GF-01 Visual Snapshot ===", rendered)
        self.assertIn("Time: t=3 (target t*=5, mode=normal)", rendered)
        self.assertIn("Timeline:", rendered)
        self.assertIn("  legend: N=now, T=target, B=now+target, edits=#interventions at t", rendered)
        self.assertIn("Legacy atom budget remaining: 4", rendered)
        self.assertIn("Budget remaining: timesteps=2", rendered)
        self.assertIn("Outputs y_t: out0=1 out1=0", rendered)
        self.assertIn("Interventions so far:", rendered)
        self.assertIn("  t=0: in0=1, in1=0", rendered)
        self.assertIn("  t=2: in0=1, in1=0", rendered)
        self.assertIn("OBS_JSON=", rendered)

        lines = rendered.splitlines()
        mark_line = next(line for line in lines if line.startswith("  mark:"))
        edits_line = next(line for line in lines if line.startswith("  edits:"))
        self.assertIn(" N", mark_line)
        self.assertIn(" T", mark_line)
        self.assertIn(" 2", edits_line)

    def test_visual_timeline_uses_b_marker_when_now_equals_target(self) -> None:
        obs = {
            "t": 2,
            "y_t": {"out0": 0},
            "effect_status_t": "not-triggered",
            "budget_t_remaining": 2,
            "budget_a_remaining": 2,
            "history_atoms": [[1, "in0", 1]],
            "mode": "hard",
            "t_star": 2,
        }
        rendered = render_visual(obs)
        lines = rendered.splitlines()
        mark_line = next(line for line in lines if line.startswith("  mark:"))
        self.assertIn(" B", mark_line)
        self.assertNotIn(" N", mark_line)
        self.assertNotIn(" T", mark_line)

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

    def test_visual_roundtrip_with_empty_y_t_and_history(self) -> None:
        obs = {
            "t": 0,
            "y_t": {},
            "effect_status_t": "not-triggered",
            "budget_t_remaining": 3,
            "budget_a_remaining": 5,
            "history_atoms": [],
            "mode": "normal",
            "t_star": 2,
        }
        rendered = render_visual(obs)
        self.assertIn("Outputs y_t: (none)", rendered)
        self.assertIn("Interventions so far:", rendered)
        self.assertIn("  (none)", rendered)
        parsed = parse_visual(rendered)
        self.assertEqual(parsed, obs)

    def test_visual_roundtrip_with_invalid_y_t_and_malformed_history(self) -> None:
        obs = {
            "t": 4,
            "y_t": None,
            "effect_status_t": "triggered",
            "budget_t_remaining": 1,
            "budget_a_remaining": 2,
            "history_atoms": [
                {"bad": "shape"},
                [0, "in0"],
                [1, "in1", "not-a-bit"],
            ],
            "mode": "hard",
            "t_star": 4,
        }
        rendered = render_visual(obs)
        self.assertIn("Outputs y_t: (invalid)", rendered)
        self.assertIn("Interventions so far:", rendered)
        self.assertIn("  (none)", rendered)
        self.assertIn("  edits:", rendered)
        parsed = parse_visual(rendered)
        self.assertEqual(parsed, obs)

    def test_parse_visual_prefers_obs_json_anchor_over_legacy_lines(self) -> None:
        anchor_obs = {
            "t": 7,
            "y_t": {"out0": 1},
            "effect_status_t": "triggered",
            "budget_t_remaining": 0,
            "budget_a_remaining": 1,
            "history_atoms": [[0, "in0", 1]],
            "mode": "hard",
            "t_star": 7,
        }
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={\"out0\":0}",
                "BT=3",
                "BA=9",
                "H=[]",
                f"OBS_JSON={render_json(anchor_obs)}",
            ]
        )
        parsed = parse_visual(rendered)
        self.assertEqual(parsed, anchor_obs)

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

    def test_parse_visual_rejects_incomplete_legacy_format(self) -> None:
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={\"out0\":1}",
                "BT=1",
                "BA=2",
                # H is intentionally omitted.
            ]
        )
        with self.assertRaises(ValueError):
            parse_visual(rendered)

    def test_parse_visual_rejects_malformed_legacy_json_payloads(self) -> None:
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={bad-json",
                "BT=1",
                "BA=2",
                "H=[[0,\"in0\",1]]",
            ]
        )
        with self.assertRaisesRegex(
            ValueError, "malformed legacy visual rendering format"
        ):
            parse_visual(rendered)

    def test_parse_visual_rejects_malformed_legacy_scalar_fields(self) -> None:
        rendered = "\n".join(
            [
                "T=NaN",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={\"out0\":1}",
                "BT=1",
                "BA=2",
                "H=[[0,\"in0\",1]]",
            ]
        )
        with self.assertRaisesRegex(
            ValueError, "malformed legacy visual rendering format"
        ):
            parse_visual(rendered)

    def test_parse_visual_rejects_legacy_yt_with_non_object_json(self) -> None:
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT=[1,2]",
                "BT=1",
                "BA=2",
                "H=[[0,\"in0\",1]]",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            r"legacy visual rendering field YT must decode to a JSON object, got \w+",
        ):
            parse_visual(rendered)

    def test_parse_visual_rejects_legacy_h_with_non_list_json(self) -> None:
        rendered = "\n".join(
            [
                "T=2",
                "MODE=normal",
                "TSTAR=4",
                "EFFECT=not-triggered",
                "YT={\"out0\":1}",
                "BT=1",
                "BA=2",
                "H={\"foo\":\"bar\"}",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            r"legacy visual rendering field H must decode to a JSON array, got \w+",
        ):
            parse_visual(rendered)


if __name__ == "__main__":
    unittest.main()
