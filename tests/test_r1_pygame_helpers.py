"""
Unit tests for map-first pygame helper utilities.

These tests avoid initializing pygame and only cover deterministic helper logic
used by the visual backend.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

from gf01.renderers.r1_pygame import (
    _CommandResponseTrailModel,
    _SectorPressureHistoryModel,
    _WaveStripModel,
    _canonical_exposure_payload,
    _apply_group_filter,
    _ap_group_key,
    _clamp_page_size,
    _clamp_timeline_span,
    _control_visible_pool,
    _cycle_pending_bit,
    _describe_output_delta,
    _effect_status_badge,
    _group_rows_for_controls,
    _grouped_input_aps,
    _help_overlay_lines,
    _normalize_binary_map,
    _onboarding_strip_lines,
    _objective_window_bounds,
    _observation_inspector_lines,
    _paginate_input_aps,
    _pressure_token,
    _pressure_level_from_observation,
    _format_top_pressure_summary,
    _sector_pressure_fill,
    _top_pressure_sectors,
    _objective_window_pressure_summary,
    _summarize_observed_outputs,
    _summarize_committed_action,
    _summarize_pending_interventions,
    _summarize_visible_ap_groups,
    _build_sector_board_cells,
    _sector_board_hover_summary,
    _build_timeline_minimap,
    _timeline_mark,
    _timeline_window_bounds,
    _truncate_ui_text,
    _wave_pressure_strip_state,
    _edits_token,
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

    def test_clamp_timeline_span(self) -> None:
        self.assertEqual(_clamp_timeline_span(2), 8)
        self.assertEqual(_clamp_timeline_span(24), 24)
        self.assertEqual(_clamp_timeline_span(99), 48)

    def test_truncate_ui_text(self) -> None:
        # Non-truncation when text is shorter than max_len.
        self.assertEqual(_truncate_ui_text("abc", max_len=5), "abc")
        # Truncation when text exceeds max_len.
        self.assertEqual(_truncate_ui_text("abcdef", max_len=5), "ab...")
        # Empty input should stay empty.
        self.assertEqual(_truncate_ui_text("", max_len=0), "")
        self.assertEqual(_truncate_ui_text("", max_len=3), "")
        self.assertEqual(_truncate_ui_text("", max_len=5), "")
        # max_len below internal minimum should behave as max_len == 4.
        self.assertEqual(_truncate_ui_text("abcdef", max_len=0), "a...")
        self.assertEqual(_truncate_ui_text("abcdef", max_len=1), "a...")
        self.assertEqual(_truncate_ui_text("abcdef", max_len=2), "a...")
        self.assertEqual(_truncate_ui_text("abcdef", max_len=3), "a...")
        # Boundary behavior at threshold.
        self.assertEqual(_truncate_ui_text("abcd", max_len=4), "abcd")
        self.assertEqual(_truncate_ui_text("abcd", max_len=5), "abcd")
        self.assertEqual(_truncate_ui_text("abcde", max_len=5), "abcde")

    def test_help_overlay_lines_include_core_controls(self) -> None:
        lines = _help_overlay_lines()
        self.assertGreaterEqual(len(lines), 6)
        self.assertTrue(any("Enter commit" in line for line in lines))
        self.assertTrue(any("Output delta" in line for line in lines))
        self.assertTrue(any("collapse" in line.lower() for line in lines))
        self.assertTrue(any("timeline zoom" in line for line in lines))
        self.assertTrue(any("inspector" in line.lower() for line in lines))
        self.assertTrue(any("Press H" in line for line in lines))

    def test_canonical_exposure_payload_without_observation(self) -> None:
        instance = SimpleNamespace(
            effect_ap="out0",
            t_star=7,
            mode="normal",
            budget_timestep=3,
            budget_atoms=9,
        )
        payload = _canonical_exposure_payload(
            last_obs=None,
            timestep=0,
            instance=instance,
            objective_text="Goal text",
        )
        mission = payload["mission"]
        self.assertEqual(mission["effect_ap"], "out0")
        self.assertEqual(mission["t_star"], 7)
        self.assertEqual(payload["observation"], None)

    def test_canonical_exposure_payload_with_observation_uses_canonical_keys(self) -> None:
        instance = SimpleNamespace(
            effect_ap="out0",
            t_star=5,
            mode="hard",
            budget_timestep=2,
            budget_atoms=4,
        )
        payload = _canonical_exposure_payload(
            last_obs={
                "t": 2,
                "y_t": {"out0": 1},
                "effect_status_t": "triggered",
                "budget_t_remaining": 1,
                "budget_a_remaining": 3,
                "history_atoms": [[0, "in0", 1]],
                "mode": "hard",
                "t_star": 5,
                "non_canonical": "ignore",
            },
            timestep=3,
            instance=instance,
            objective_text="Goal text",
        )
        observation = payload["observation"]
        self.assertIsInstance(observation, dict)
        self.assertIn("history_atoms", observation)
        self.assertNotIn("non_canonical", observation)

    def test_observation_inspector_lines_render_mission_and_observation_json(self) -> None:
        lines = _observation_inspector_lines(
            {
                "mission": {"effect_ap": "out0", "mode": "normal"},
                "observation": {"t": 1, "y_t": {"out0": 1}},
            }
        )
        self.assertGreaterEqual(len(lines), 3)
        self.assertIn("inspector", lines[0].lower())
        self.assertIn("mission:", lines[1])
        self.assertIn("observation:", lines[2])

    def test_observation_inspector_lines_handles_none_observation(self) -> None:
        lines = _observation_inspector_lines(
            {
                "mission": {"effect_ap": "out0", "mode": "normal"},
                "observation": None,
            }
        )
        self.assertGreaterEqual(len(lines), 3)
        self.assertIn("inspector", lines[0].lower())
        self.assertIn("mission:", lines[1])
        self.assertIn("observation: (none yet)", lines[2].lower())

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

    def test_wave_pressure_strip_state_awaiting_observation(self) -> None:
        label, filled, _, trend = _wave_pressure_strip_state(None, {})
        self.assertIn("awaiting observation", label)
        self.assertEqual(filled, 0)
        self.assertEqual(trend, "none")

    def test_wave_pressure_strip_state_baseline(self) -> None:
        label, filled, _, trend = _wave_pressure_strip_state(
            None,
            {"out0": 1, "out1": 0, "out2": 1, "out3": 0},
        )
        self.assertIn("baseline", label)
        self.assertEqual(filled, 5)
        self.assertEqual(trend, "baseline")

    def test_wave_pressure_strip_state_rising(self) -> None:
        label, _, _, trend = _wave_pressure_strip_state(
            {"out0": 0, "out1": 0, "out2": 0, "out3": 0},
            {"out0": 1, "out1": 1, "out2": 0, "out3": 0},
        )
        self.assertIn("rising", label)
        self.assertEqual(trend, "rising")

    def test_pressure_level_from_observation(self) -> None:
        self.assertIsNone(_pressure_level_from_observation(None))
        self.assertIsNone(_pressure_level_from_observation({}))
        self.assertEqual(
            _pressure_level_from_observation({"out0": 0, "out1": 0, "out2": 0}),
            0,
        )
        self.assertEqual(
            _pressure_level_from_observation({"out0": 1, "out1": 0, "out2": 1, "out3": 0}),
            5,
        )
        self.assertEqual(
            _pressure_level_from_observation({"out0": 1, "out1": 1, "out2": 1}),
            10,
        )

    def test_sector_pressure_fill_clamps(self) -> None:
        low = _sector_pressure_fill(-99)
        mid = _sector_pressure_fill(5)
        high = _sector_pressure_fill(99)
        self.assertEqual(len(low), 3)
        self.assertEqual(len(mid), 3)
        self.assertEqual(len(high), 3)
        self.assertTrue(sum(low) < sum(mid) < sum(high))

    def test_sector_pressure_history_model_records_and_limits(self) -> None:
        model = _SectorPressureHistoryModel(max_entries=2)
        model.record(timestep=0, y_t={"out0": 0, "out1": 0})
        model.record(timestep=2, y_t={"out0": 1, "out1": 1})
        model.record(timestep=1, y_t={"out0": 1, "out1": 0})
        self.assertEqual(model.levels(), {1: 5, 2: 10})
        model.record(timestep=3, y_t={})
        self.assertEqual(model.levels(), {1: 5, 2: 10})
        model.reset()
        self.assertEqual(model.levels(), {})

    def test_pressure_token(self) -> None:
        self.assertEqual(_pressure_token(None), "P.")
        self.assertEqual(_pressure_token(-4), "P0")
        self.assertEqual(_pressure_token(3), "P3")
        self.assertEqual(_pressure_token(999), "P10")

    def test_edits_token(self) -> None:
        self.assertEqual(_edits_token(None), "E.")
        self.assertEqual(_edits_token(-5), "E0")
        self.assertEqual(_edits_token(0), "E0")
        self.assertEqual(_edits_token(4), "E4")

    def test_top_pressure_sectors_ranking(self) -> None:
        ranked = _top_pressure_sectors(
            {
                0: 4,
                1: 7,
                2: 7,
                3: 2,
                4: 10,
            },
            max_items=3,
        )
        self.assertEqual(ranked, [(4, 10), (2, 7), (1, 7)])

    def test_format_top_pressure_summary(self) -> None:
        self.assertEqual(_format_top_pressure_summary({}, max_items=3), "(none yet)")
        self.assertEqual(
            _format_top_pressure_summary({3: 5, 4: 7, 2: 7}, max_items=2),
            "t=4:P7, t=2:P7",
        )

    def test_objective_window_pressure_summary(self) -> None:
        self.assertEqual(
            _objective_window_pressure_summary(
                pressure_levels={},
                window_start=5,
                window_end=7,
            ),
            "Window pressure: 0/3 observed",
        )
        self.assertEqual(
            _objective_window_pressure_summary(
                pressure_levels={5: 2, 6: 9, 8: 10},
                window_start=5,
                window_end=7,
            ),
            "Window pressure: 2/3 observed, peak t=6:P9, avg=5.5",
        )

    def test_wave_strip_model_trail_dedup_and_window(self) -> None:
        model = _WaveStripModel()
        self.assertEqual(model.trail_text(), "(none yet)")
        model.update_history(timestep=0, trend="none")
        self.assertEqual(model.trail_text(), "(none yet)")
        model.update_history(timestep=1, trend="baseline")
        model.update_history(timestep=1, trend="baseline")
        model.update_history(timestep=2, trend="rising")
        model.update_history(timestep=3, trend="steady")
        model.update_history(timestep=4, trend="falling")
        model.update_history(timestep=5, trend="rising")
        trail = model.trail_text()
        self.assertIn("t=2:rising", trail)
        self.assertIn("t=5:rising", trail)
        self.assertNotIn("t=1:baseline | t=1:baseline", trail)

    def test_command_response_trail_empty(self) -> None:
        model = _CommandResponseTrailModel()
        self.assertEqual(model.lines(), ["(none yet)"])

    def test_command_response_trail_dedup_and_window(self) -> None:
        model = _CommandResponseTrailModel(max_entries=3, max_visible_lines=3)
        model.record(timestep=1, command="in0=1", response_delta="d0")
        model.record(timestep=1, command="in0=1", response_delta="d0")
        model.record(timestep=2, command="in0=0", response_delta="d1")
        model.record(timestep=3, command="skip", response_delta="d2")
        model.record(timestep=4, command="in2=1", response_delta="d3")
        self.assertEqual(
            model.lines(),
            [
                "t=2 | in0=0 -> d1",
                "t=3 | skip -> d2",
                "t=4 | in2=1 -> d3",
            ],
        )

    def test_command_response_trail_respects_visible_window(self) -> None:
        model = _CommandResponseTrailModel(max_entries=5, max_visible_lines=2)
        model.record(timestep=1, command="in0=1", response_delta="d0")
        model.record(timestep=2, command="in0=0", response_delta="d1")
        model.record(timestep=3, command="skip", response_delta="d2")
        self.assertEqual(
            model.lines(),
            ["t=2 | in0=0 -> d1", "t=3 | skip -> d2"],
        )

    def test_command_response_trail_truncates_long_lines(self) -> None:
        model = _CommandResponseTrailModel(
            max_entries=4,
            max_visible_lines=3,
            line_max_len=18,
        )
        model.record(
            timestep=9,
            command="x" * 20,
            response_delta="y" * 20,
        )
        lines = model.lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], "t=9 | xxxxxxxxx...")
        self.assertTrue(lines[0].endswith("..."))

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

    def test_timeline_window_bounds_short_horizon(self) -> None:
        start, end = _timeline_window_bounds(
            timestep=3,
            t_star=7,
            history_counts={1: 2, 5: 1},
            span=24,
        )
        self.assertEqual((start, end), (0, 7))

    def test_timeline_window_bounds_includes_now_and_target_when_possible(self) -> None:
        start, end = _timeline_window_bounds(
            timestep=10,
            t_star=18,
            history_counts={30: 1},
            span=16,
        )
        self.assertLessEqual(start, 10)
        self.assertGreaterEqual(end, 10)
        self.assertLessEqual(start, 18)
        self.assertGreaterEqual(end, 18)
        self.assertEqual(end - start + 1, 16)

    def test_timeline_window_bounds_prioritizes_now_when_span_too_small(self) -> None:
        start, end = _timeline_window_bounds(
            timestep=5,
            t_star=45,
            history_counts={45: 2},
            span=12,
        )
        self.assertLessEqual(start, 5)
        self.assertGreaterEqual(end, 5)
        self.assertEqual(end - start + 1, 12)

    def test_build_timeline_minimap_contains_window_and_markers(self) -> None:
        minimap = _build_timeline_minimap(
            max_t=20,
            start_t=4,
            end_t=12,
            timestep=7,
            t_star=15,
            window_start=10,
            window_end=15,
            history_counts={3: 1, 8: 2},
            pressure_levels={5: 3, 11: 7},
            width=32,
        )
        self.assertEqual(len(minimap), 32)
        self.assertIn("[", minimap)
        self.assertIn("]", minimap)
        self.assertIn("N", minimap)
        self.assertIn("T", minimap)

    def test_build_timeline_minimap_marks_shared_now_target_with_b(self) -> None:
        minimap = _build_timeline_minimap(
            max_t=10,
            start_t=0,
            end_t=5,
            timestep=4,
            t_star=4,
            window_start=2,
            window_end=6,
            history_counts={},
            pressure_levels={},
            width=24,
        )
        self.assertIn("B", minimap)

    def test_build_timeline_minimap_horizon_zero_keeps_marker_visible(self) -> None:
        minimap = _build_timeline_minimap(
            max_t=0,
            start_t=0,
            end_t=0,
            timestep=0,
            t_star=0,
            window_start=0,
            window_end=0,
            history_counts={},
            pressure_levels={},
            width=16,
        )
        self.assertEqual(len(minimap), 16)
        self.assertIn("[", minimap)
        self.assertIn("]", minimap)
        self.assertIn("B", minimap)

    def test_build_sector_board_cells_assigns_markers_and_flags(self) -> None:
        cells = _build_sector_board_cells(
            max_t=23,
            timestep=5,
            t_star=18,
            start_t=4,
            end_t=10,
            window_start=16,
            window_end=20,
            history_counts={5: 1, 6: 2, 18: 1},
            pressure_levels={4: 3, 8: 7, 18: 9},
            cols=8,
            rows=6,
        )
        self.assertEqual(len(cells), 48)
        self.assertTrue(any(cell.marker == "N" for cell in cells))
        self.assertTrue(any(cell.marker == "T" for cell in cells))
        self.assertTrue(any(cell.in_viewport for cell in cells))
        self.assertTrue(any(cell.in_objective_window for cell in cells))

    def test_build_sector_board_cells_clamps_pressure_levels(self) -> None:
        cells = _build_sector_board_cells(
            max_t=7,
            timestep=3,
            t_star=3,
            start_t=0,
            end_t=7,
            window_start=0,
            window_end=7,
            history_counts={},
            pressure_levels={3: 99},
            cols=4,
            rows=2,
        )
        levels = [cell.pressure_level for cell in cells if cell.pressure_level is not None]
        self.assertTrue(levels)
        self.assertTrue(all(level <= 10 for level in levels))
        self.assertTrue(any(cell.marker == "B" for cell in cells))

    def test_sector_board_hover_summary_for_cell(self) -> None:
        cell = _build_sector_board_cells(
            max_t=7,
            timestep=2,
            t_star=5,
            start_t=0,
            end_t=7,
            window_start=4,
            window_end=6,
            history_counts={2: 1},
            pressure_levels={2: 6},
            cols=4,
            rows=2,
        )[2]
        summary = _sector_board_hover_summary(cell)
        self.assertIn("t=", summary)
        self.assertIn("P", summary)
        self.assertIn("E", summary)

    def test_sector_board_hover_summary_without_cell(self) -> None:
        self.assertIn("Hover", _sector_board_hover_summary(None))

    def test_objective_window_bounds_hard_mode(self) -> None:
        self.assertEqual(
            _objective_window_bounds(mode="hard", t_star=12, window_size=4),
            (12, 12),
        )

    def test_objective_window_bounds_hard_mode_normalization_variants(self) -> None:
        canonical = _objective_window_bounds(mode="hard", t_star=12, window_size=4)
        variants = ["HARD", " Hard ", " HARD  ", "HaRd", "\thard\n"]
        for variant in variants:
            with self.subTest(mode_variant=variant):
                self.assertEqual(
                    _objective_window_bounds(
                        mode=variant, t_star=12, window_size=4
                    ),
                    canonical,
                )

    def test_objective_window_bounds_normal_mode(self) -> None:
        self.assertEqual(
            _objective_window_bounds(mode="normal", t_star=12, window_size=4),
            (8, 12),
        )

    def test_objective_window_bounds_clamps_at_zero(self) -> None:
        self.assertEqual(
            _objective_window_bounds(mode="normal", t_star=2, window_size=5),
            (0, 2),
        )


if __name__ == "__main__":
    unittest.main()
