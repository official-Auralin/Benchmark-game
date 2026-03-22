"""Spatial causal-board helpers for the GF-01 visual renderer.

The board is no longer a disguised timeline. It is a deterministic spatial
layout derived from proposition roles and coarse relation structure in the
formal task, while time remains represented separately in the mission/timeline
widgets.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..semantics import run_automaton
from .r1_theme import tile_color

if TYPE_CHECKING:
    from ..models import GF01Instance


GRID_COLS = 6
GRID_ROWS = 4
TILE_W = 96
TILE_H = 48


@dataclass(frozen=True)
class TileData:
    row: int
    col: int
    tile_type: str
    base_color: tuple[int, int, int]
    is_objective: bool
    is_current_wave: bool
    defense_ap: str | None
    output_ap: str | None
    role: str
    label: str | None
    defense_index: int | None
    threat_level: float
    has_edits: bool
    deployed_value: int | None
    iso_x: int
    iso_y: int


def iso_project(col: int, row: int, *, origin_x: int, origin_y: int) -> tuple[int, int]:
    ix = origin_x + (col - row) * (TILE_W // 2)
    iy = origin_y + (col + row) * (TILE_H // 2)
    return ix, iy


def relation_weights(instance: "GF01Instance") -> dict[tuple[str, str], int]:
    baseline_outputs = run_automaton(instance.automaton, instance.base_trace)[1]
    weights: dict[tuple[str, str], int] = {
        (input_ap, output_ap): 0
        for input_ap in instance.automaton.input_aps
        for output_ap in instance.automaton.output_aps
    }
    for t, step in enumerate(instance.base_trace):
        for input_ap in instance.automaton.input_aps:
            flipped_trace = [{k: int(v) for k, v in row.items()} for row in instance.base_trace]
            flipped_trace[t][input_ap] = 1 - int(step[input_ap])
            changed_outputs = run_automaton(instance.automaton, flipped_trace)[1]
            for output_ap in instance.automaton.output_aps:
                delta = 0
                for idx in range(t, len(baseline_outputs)):
                    if (
                        int(changed_outputs[idx].get(output_ap, 0))
                        != int(baseline_outputs[idx].get(output_ap, 0))
                    ):
                        delta += 1
                weights[(input_ap, output_ap)] += delta
    return weights


def grid_dimensions(instance: "GF01Instance") -> tuple[int, int]:
    input_count = len(instance.automaton.input_aps)
    output_count = len(instance.automaton.output_aps)
    rel_count = sum(1 for value in relation_weights(instance).values() if value > 0)
    cols = max(5, min(10, 5 + int(rel_count > (input_count + output_count))))
    output_slot_cols = 1 if cols < 6 else 2
    rows = max(
        3,
        math.ceil(input_count / 2),
        math.ceil(output_count / output_slot_cols),
    )
    return cols, rows


def _rank_inputs(instance: "GF01Instance") -> list[str]:
    weights = relation_weights(instance)
    target = instance.effect_ap
    return sorted(
        instance.automaton.input_aps,
        key=lambda ap: (
            -weights.get((ap, target), 0),
            -sum(weights.get((ap, out), 0) for out in instance.automaton.output_aps),
            ap,
        ),
    )


def _rank_outputs(instance: "GF01Instance") -> list[str]:
    weights = relation_weights(instance)
    return sorted(
        instance.automaton.output_aps,
        key=lambda output_ap: (
            0 if output_ap == instance.effect_ap else 1,
            -sum(weights.get((inp, output_ap), 0) for inp in instance.automaton.input_aps),
            output_ap,
        ),
    )


def _spread_positions(items: list[str], columns: list[int], rows: int) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    for idx, item in enumerate(items):
        col = columns[idx % len(columns)]
        band = idx // len(columns)
        row = min(rows - 1, band)
        positions[item] = (row, col)
    return positions


def build_grid(
    instance: "GF01Instance",
    *,
    timestep: int,
    pressure_levels: Mapping[int, int] | None = None,
    history_counts: Mapping[int, int] | None = None,
    history_atoms: list[tuple[int, str, int]] | None = None,
    current_outputs: Mapping[str, int] | None = None,
    origin_x: int = 0,
    origin_y: int = 0,
) -> list[TileData]:
    cols, rows = grid_dimensions(instance)
    input_aps = _rank_inputs(instance)
    output_aps = _rank_outputs(instance)
    history_by_ap: dict[str, int] = {}
    for _t, ap_name, value in history_atoms or []:
        history_by_ap[ap_name] = int(value)

    input_positions = _spread_positions(input_aps, [0, 1], rows)
    output_columns = [cols - 1] if cols < 6 else [cols - 1, cols - 2]
    output_positions = _spread_positions(output_aps, output_columns, rows)

    current_outputs = {str(k): int(v) for k, v in (current_outputs or {}).items()}
    history_counts = history_counts or {}
    _ = (pressure_levels, timestep)

    tiles: list[TileData] = []
    for row in range(rows):
        for col in range(cols):
            defense_ap = next((ap for ap, pos in input_positions.items() if pos == (row, col)), None)
            output_ap = next((ap for ap, pos in output_positions.items() if pos == (row, col)), None)
            role = "neutral"
            label: str | None = None
            is_objective = False
            defense_index: int | None = None
            threat_level = 0.0
            deployed_value: int | None = None
            has_edits = False
            tile_type = "neutral"

            if defense_ap is not None:
                role = "input"
                label = defense_ap
                tile_type = "input"
                defense_index = instance.automaton.input_aps.index(defense_ap)
                deployed_value = history_by_ap.get(defense_ap)
                has_edits = defense_ap in history_by_ap
            elif output_ap is not None:
                role = "output"
                label = output_ap
                tile_type = "output"
                is_objective = output_ap == instance.effect_ap
                threat_level = float(current_outputs.get(output_ap, 0))
                has_edits = any(count > 0 for count in history_counts.values())
            else:
                input_rows = {pos[0] for pos in input_positions.values()}
                output_rows = {pos[0] for pos in output_positions.values()}
                if row in input_rows or row in output_rows:
                    tile_type = "bridge"
                    role = "bridge"

            ix, iy = iso_project(col, row, origin_x=origin_x, origin_y=origin_y)
            tiles.append(
                TileData(
                    row=row,
                    col=col,
                    tile_type=tile_type,
                    base_color=tile_color("objective" if is_objective else tile_type),
                    is_objective=is_objective,
                    is_current_wave=False,
                    defense_ap=defense_ap,
                    output_ap=output_ap,
                    role=role,
                    label=label,
                    defense_index=defense_index,
                    threat_level=threat_level,
                    has_edits=has_edits,
                    deployed_value=deployed_value,
                    iso_x=ix,
                    iso_y=iy,
                )
            )
    return tiles


def grid_bounding_box(tiles: list[TileData]) -> tuple[int, int, int, int]:
    if not tiles:
        return (0, 0, 0, 0)
    xs = [t.iso_x for t in tiles]
    ys = [t.iso_y for t in tiles]
    return (min(xs), min(ys), max(xs) + TILE_W, max(ys) + TILE_H)


def tile_at_screen_pos(tiles: list[TileData], mx: int, my: int) -> TileData | None:
    for tile in reversed(tiles):
        cx = tile.iso_x + TILE_W // 2
        cy = tile.iso_y + TILE_H // 2
        dx = abs(mx - cx) / (TILE_W / 2.0)
        dy = abs(my - cy) / (TILE_H / 2.0)
        if dx + dy <= 1.0:
            return tile
    return None


def wave_timeline_data(
    total_waves: int,
    current: int,
    t_star: int,
    mode: str,
    window_size: int,
    pressure_levels: Mapping[int, int] | None = None,
    history_counts: Mapping[int, int] | None = None,
) -> list[dict[str, object]]:
    window_start = t_star if mode == "hard" else max(0, t_star - window_size)
    obs_pressure = pressure_levels or {}
    obs_edits = history_counts or {}
    entries: list[dict[str, object]] = []
    for t in range(total_waves):
        entries.append(
            {
                "wave": t + 1,
                "t": t,
                "is_current": t == current,
                "is_critical": t == t_star,
                "in_window": window_start <= t <= t_star,
                "pressure": obs_pressure.get(t),
                "has_edits": obs_edits.get(t, 0) > 0,
                "is_past": t < current,
            }
        )
    return entries


def defense_tile_assignments(
    input_aps: list[str],
    total_cells: int,
    seed: int,
) -> dict[int, str]:
    _ = seed
    assignments: dict[int, str] = {}
    if not input_aps:
        return assignments
    stride = max(1, total_cells // max(1, len(input_aps)))
    for idx, ap in enumerate(input_aps):
        cell_idx = min(total_cells - 1, idx * stride)
        while cell_idx in assignments and cell_idx < total_cells - 1:
            cell_idx += 1
        assignments[cell_idx] = ap
    return assignments
