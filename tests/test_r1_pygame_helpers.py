"""
Unit tests for map-first pygame helper utilities.

These tests avoid initializing pygame and only cover deterministic helper logic
used by the visual backend.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

import unittest

from gf01.renderers.r1_pygame import (
    _apply_group_filter,
    _ap_group_key,
    _clamp_page_size,
    _control_visible_pool,
    _cycle_pending_bit,
    _describe_output_delta,
    _effect_status_badge,
    _group_rows_for_controls,
    _grouped_input_aps,
    _help_overlay_lines,
    _normalize_binary_map,
    _onboarding_strip_lines,
    _paginate_input_aps,
    _summarize_observed_outputs,
    _summarize_committed_action,
    _summarize_pending_interventions,
    _summarize_visible_ap_groups,
    _timeline_mark,
)


class TestR1PygameHelpers(unittest.TestCase):
    def test_paginate_input_aps_empty(self) -> None:
        visible, page, total_pages = _paginate_input_aps([], page=3, page_size=10)
        self.assertEqual(visible, [])
        self.assertEqual(page, 0)
        self.assertEqual(total_pages, 1)

    def test_paginate_input_aps_clamps_page(self) -> None:
        aps = [f"in{i}" for i in range(23)]
        visible, page, total_pages = _paginate_input_aps(aps, page=99, page_size=10)
        self.assertEqual(total_pages, 3)
        self.assertEqual(page, 2)
        self.assertEqual(visible, ["in20", "in21", "in22"])

    def test_paginate_input_aps_mid_page(self) -> None:
        aps = [f"in{i}" for i in range(23)]
        visible, page, total_pages = _paginate_input_aps(aps, page=1, page_size=10)
        self.assertEqual(total_pages, 3)
        self.assertEqual(page, 1)
        self.assertEqual(visible, [f"in{i}" for i in range(10, 20)])

    def test_paginate_input_aps_rejects_nonpositive_page_size(self) -> None:
        with self.assertRaises(ValueError):
            _paginate_input_aps(["in0"], page=0, page_size=0)

    def test_normalize_binary_map_filters_invalid_entries(self) -> None:
        normalized = _normalize_binary_map(
            {
                "out0": 1,
                "out1": 0,
                "bad_text": "x",
                "bad_value": 2,
            }
        )
        self.assertEqual(normalized, {"out0": 1, "out1": 0})

    def test_describe_output_delta_baseline(self) -> None:
        delta = _describe_output_delta(None, {"out0": 1})
        self.assertEqual(delta, "Output delta: baseline observation")

    def test_describe_output_delta_no_change(self) -> None:
        delta = _describe_output_delta({"out0": 1, "out1": 0}, {"out0": 1, "out1": 0})
        self.assertEqual(delta, "Output delta: no observed change")

    def test_describe_output_delta_with_changes(self) -> None:
        delta = _describe_output_delta({"out0": 0, "out1": 1}, {"out0": 1, "out1": 1})
        self.assertIn("out0: 0 -> 1", delta)

    def test_cycle_pending_bit_cycles_1_to_0_to_clear(self) -> None:
        pending: dict[str, int] = {}
        _cycle_pending_bit(pending, "in0")
        self.assertEqual(pending, {"in0": 1})
        _cycle_pending_bit(pending, "in0")
        self.assertEqual(pending, {"in0": 0})
        _cycle_pending_bit(pending, "in0")
        self.assertEqual(pending, {})

    def test_clamp_page_size(self) -> None:
        self.assertEqual(_clamp_page_size(2), 4)
        self.assertEqual(_clamp_page_size(10), 10)
        self.assertEqual(_clamp_page_size(20), 16)

    def test_help_overlay_lines_include_core_controls(self) -> None:
        lines = _help_overlay_lines()
        self.assertGreaterEqual(len(lines), 6)
        self.assertTrue(any("Enter commit" in line for line in lines))
        self.assertTrue(any("Output delta" in line for line in lines))
        self.assertTrue(any("collapse" in line.lower() for line in lines))
        self.assertTrue(any("Press H" in line for line in lines))

    def test_ap_group_key(self) -> None:
        self.assertEqual(_ap_group_key("in0"), "in")
        self.assertEqual(_ap_group_key("sensor_temp_1"), "sensor_temp")
        self.assertEqual(_ap_group_key("x"), "x")

    def test_summarize_visible_ap_groups(self) -> None:
        summary = _summarize_visible_ap_groups(["in0", "in1", "mode0", "sensor0"])
        self.assertIn("in(2)", summary)
        self.assertIn("mode(1)", summary)
        self.assertIn("sensor(1)", summary)

    def test_summarize_observed_outputs_active_and_total(self) -> None:
        summary = _summarize_observed_outputs({"out0": 1, "out1": 0, "out2": 1})
        self.assertIn("Observed outputs ON:", summary)
        self.assertIn("(2/3)", summary)
        self.assertIn("out0", summary)
        self.assertIn("out2", summary)

    def test_summarize_observed_outputs_all_off(self) -> None:
        summary = _summarize_observed_outputs({"out0": 0, "out1": 0})
        self.assertEqual(summary, "Observed outputs: all OFF (0/2 ON)")

    def test_summarize_pending_interventions(self) -> None:
        summary = _summarize_pending_interventions({"in2": 0, "in0": 1})
        self.assertTrue(summary.startswith("Pending interventions:"))
        self.assertIn("in0=1", summary)
        self.assertIn("in2=0", summary)

    def test_summarize_pending_interventions_none(self) -> None:
        self.assertEqual(
            _summarize_pending_interventions({}),
            "Pending interventions: none selected",
        )

    def test_summarize_committed_action(self) -> None:
        summary = _summarize_committed_action({"in2": 0, "in0": 1})
        self.assertEqual(summary, "in0=1, in2=0")

    def test_summarize_committed_action_skip(self) -> None:
        self.assertEqual(
            _summarize_committed_action({}),
            "no interventions (skip)",
        )

    def test_effect_status_badge_triggered(self) -> None:
        text, _ = _effect_status_badge("triggered")
        self.assertEqual(text, "Objective active")

    def test_effect_status_badge_not_triggered(self) -> None:
        text, _ = _effect_status_badge("not-triggered")
        self.assertEqual(text, "Objective not active")

    def test_effect_status_badge_fallback(self) -> None:
        text, _ = _effect_status_badge("custom")
        self.assertIn("Objective status:", text)

    def test_onboarding_strip_lines_for_first_three_steps(self) -> None:
        self.assertTrue(_onboarding_strip_lines(0))
        self.assertTrue(_onboarding_strip_lines(1))
        self.assertTrue(_onboarding_strip_lines(2))

    def test_onboarding_strip_lines_after_step_two_empty(self) -> None:
        self.assertEqual(_onboarding_strip_lines(3), [])

    def test_grouped_input_aps(self) -> None:
        grouped = _grouped_input_aps(["in0", "in1", "sensor0", "mode0"])
        self.assertEqual(grouped["in"], ["in0", "in1"])
        self.assertEqual(grouped["mode"], ["mode0"])
        self.assertEqual(grouped["sensor"], ["sensor0"])

    def test_apply_group_filter(self) -> None:
        aps = ["in0", "in1", "sensor0", "mode0"]
        self.assertEqual(_apply_group_filter(aps, None), aps)
        self.assertEqual(_apply_group_filter(aps, "in"), ["in0", "in1"])
        self.assertEqual(_apply_group_filter(aps, "unknown"), [])

    def test_control_visible_pool(self) -> None:
        aps = ["in0", "in1", "sensor0", "mode0"]
        self.assertEqual(
            _control_visible_pool(aps, group_filter=None, collapse_rows=False),
            aps,
        )
        self.assertEqual(
            _control_visible_pool(aps, group_filter="in", collapse_rows=False),
            ["in0", "in1"],
        )
        self.assertEqual(
            _control_visible_pool(aps, group_filter=None, collapse_rows=True),
            [],
        )
        self.assertEqual(
            _control_visible_pool(aps, group_filter="sensor", collapse_rows=True),
            ["sensor0"],
        )

    def test_group_rows_for_controls(self) -> None:
        aps = ["in0", "in1", "sensor0", "mode0"]
        pending = {"in0": 1, "sensor0": 0}
        lines = _group_rows_for_controls(aps, pending, max_groups=8)
        self.assertIn("in: 1/2 set", lines)
        self.assertIn("mode: 0/1 set", lines)
        self.assertIn("sensor: 1/1 set", lines)

    def test_timeline_mark(self) -> None:
        self.assertEqual(_timeline_mark(2, timestep=2, t_star=5), "N")
        self.assertEqual(_timeline_mark(5, timestep=2, t_star=5), "T")
        self.assertEqual(_timeline_mark(3, timestep=3, t_star=3), "B")
        self.assertEqual(_timeline_mark(1, timestep=3, t_star=5), "")


if __name__ == "__main__":
    unittest.main()
