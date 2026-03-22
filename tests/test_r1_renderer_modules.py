from __future__ import annotations

import unittest

from gf01.models import GF01Instance, MealyAutomaton
from gf01.renderers.r1_grid import (
    GRID_COLS,
    GRID_ROWS,
    TILE_H,
    TILE_W,
    build_grid,
    defense_tile_assignments,
    grid_dimensions,
    iso_project,
    tile_at_screen_pos,
    wave_timeline_data,
)
from gf01.renderers.r1_pygame import (
    _canonical_inspector_payload,
    _inspector_lines,
    _normalize_binary_map,
)
from gf01.renderers.r1_theme import (
    effect_status_display,
    objective_text_themed,
    threat_name,
    tile_color,
    tile_type_for_index,
    victory_text,
    defense_color,
    defense_icon,
    defense_name,
)
from gf01.semantics import input_key


def _toy_instance(mode: str = "normal") -> GF01Instance:
    input_aps = ["in0", "in1", "in2"]
    output_aps = ["out0", "out1"]
    transitions: dict[str, dict[str, tuple[str, dict[str, int]]]] = {"s0": {}}
    for b0 in (0, 1):
        for b1 in (0, 1):
            for b2 in (0, 1):
                valuation = {"in0": b0, "in1": b1, "in2": b2}
                transitions["s0"][input_key(input_aps, valuation)] = (
                    "s0",
                    {
                        "out0": b0 or b2,
                        "out1": b1,
                    },
                )
    automaton = MealyAutomaton(
        states=["s0"],
        initial_state="s0",
        input_aps=input_aps,
        output_aps=output_aps,
        transitions=transitions,
    )
    return GF01Instance(
        instance_id=f"toy-{mode}",
        automaton=automaton,
        base_trace=[{"in0": 0, "in1": 0, "in2": 0} for _ in range(6)],
        effect_ap="out0",
        t_star=4,
        mode=mode,
        window_size=2,
        budget_timestep=3,
        budget_atoms=5,
        seed=7,
    )


class TestR1Grid(unittest.TestCase):
    def test_defense_tile_assignments_are_deterministic(self) -> None:
        input_aps = ["in0", "in1", "in2", "in3"]
        first = defense_tile_assignments(input_aps, GRID_COLS * GRID_ROWS, 7)
        second = defense_tile_assignments(input_aps, GRID_COLS * GRID_ROWS, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), len(input_aps))

    def test_build_grid_marks_objective_current_and_deployments(self) -> None:
        instance = _toy_instance("normal")
        cols, rows = grid_dimensions(instance)
        tiles = build_grid(
            instance,
            timestep=3,
            pressure_levels={1: 7, 3: 9},
            history_counts={1: 1, 3: 2},
            history_atoms=[(1, "in0", 1), (3, "in1", 0)],
            current_outputs={"out0": 1, "out1": 0},
            origin_x=120,
            origin_y=160,
        )
        self.assertEqual(len(tiles), cols * rows)
        objective_tile = next(tile for tile in tiles if tile.is_objective)
        self.assertEqual(objective_tile.output_ap, "out0")
        self.assertFalse(any(tile.is_current_wave for tile in tiles))
        self.assertTrue(any(tile.has_edits for tile in tiles if tile.defense_ap is not None))
        deployed_tile = next(tile for tile in tiles if tile.defense_ap == "in0")
        self.assertEqual(deployed_tile.deployed_value, 1)

    def test_iso_project_and_tile_at_screen_pos_round_trip(self) -> None:
        instance = _toy_instance("normal")
        origin_x, origin_y = 220, 140
        tiles = build_grid(instance, timestep=0, origin_x=origin_x, origin_y=origin_y)
        first = tiles[0]
        self.assertEqual((first.iso_x, first.iso_y), iso_project(0, 0, origin_x=origin_x, origin_y=origin_y))
        center_x = first.iso_x + TILE_W // 2
        center_y = first.iso_y + TILE_H // 2
        hit = tile_at_screen_pos(tiles, center_x, center_y)
        self.assertIsNotNone(hit)
        self.assertEqual((hit.row, hit.col), (first.row, first.col))

    def test_wave_timeline_data_tracks_window_pressure_and_edits(self) -> None:
        entries = wave_timeline_data(
            total_waves=6,
            current=3,
            t_star=4,
            mode="normal",
            window_size=2,
            pressure_levels={2: 5, 3: 9},
            history_counts={2: 1},
        )
        self.assertEqual(len(entries), 6)
        self.assertTrue(entries[3]["is_current"])
        self.assertTrue(entries[4]["is_critical"])
        self.assertTrue(entries[2]["in_window"])
        self.assertEqual(entries[2]["pressure"], 5)
        self.assertTrue(entries[2]["has_edits"])


class TestR1Theme(unittest.TestCase):
    def test_defense_theme_mapping_is_deterministic(self) -> None:
        input_aps = ["in0", "in1", "in2"]
        self.assertEqual(defense_name("in1", input_aps), defense_name("in1", input_aps))
        self.assertEqual(defense_icon("in2", input_aps), defense_icon("in2", input_aps))
        self.assertEqual(defense_color("in0", input_aps), defense_color("in0", input_aps))

    def test_objective_text_themed_reflects_mode(self) -> None:
        normal = objective_text_themed(_toy_instance("normal"))
        hard = objective_text_themed(_toy_instance("hard"))
        self.assertIn("Cause", normal)
        self.assertIn("steps", normal)
        self.assertIn("exactly step", hard)

    def test_effect_status_display_and_victory_text(self) -> None:
        self.assertEqual(effect_status_display("triggered")[0], "TARGET TRIGGERED")
        self.assertEqual(effect_status_display("not-triggered")[0], "TARGET NOT TRIGGERED")
        self.assertEqual(victory_text(True, True, True)[0], "CERTIFIED")
        self.assertEqual(victory_text(False, False, False)[0], "GOAL MISSED")

    def test_tile_palette_and_threat_name_are_stable(self) -> None:
        self.assertEqual(tile_type_for_index(3, 48, 7), tile_type_for_index(3, 48, 7))
        self.assertEqual(tile_color("objective"), (154, 114, 52))
        self.assertEqual(threat_name("out0", ["out0", "out1"]), "Out0")


class TestR1PygamePureHelpers(unittest.TestCase):
    def test_normalize_binary_map_filters_invalid_entries(self) -> None:
        normalized = _normalize_binary_map({"out0": 1, "out1": 0, "bad": 3, "text": "x"})
        self.assertEqual(normalized, {"out0": 1, "out1": 0})

    def test_canonical_inspector_payload_filters_to_canonical_keys(self) -> None:
        instance = _toy_instance("normal")
        payload = _canonical_inspector_payload(
            last_obs={
                "t": 2,
                "y_t": {"out0": 1},
                "effect_status_t": "triggered",
                "budget_t_remaining": 1,
                "budget_a_remaining": 2,
                "history_atoms": [[1, "in0", 1]],
                "mode": "normal",
                "t_star": 4,
                "non_canonical": "ignore",
            },
            instance=instance,
        )
        observation = payload["observation"]
        self.assertIsInstance(observation, dict)
        self.assertIn("history_atoms", observation)
        self.assertNotIn("non_canonical", observation)

    def test_inspector_lines_handle_missing_observation(self) -> None:
        lines = _inspector_lines(
            _canonical_inspector_payload(
                last_obs=None,
                instance=_toy_instance("normal"),
            )
        )
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(lines[0], "CANONICAL OBSERVATION")
        self.assertIn("mission:", lines[1])
        self.assertTrue(any("observation: (none yet)" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
