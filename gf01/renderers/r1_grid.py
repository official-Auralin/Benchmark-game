"""Grid and tile-map helpers for the canonical GF-01-R1 tower-defense renderer.

Builds the grid layout, assigns tile types, computes isometric projection
coordinates, and determines which tiles carry defense icons or threat overlays.
No pygame import -- pure geometry and data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .r1_theme import tile_type_for_index, tile_color

if TYPE_CHECKING:
    from ..models import GF01Instance


GRID_COLS = 8
GRID_ROWS = 6
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
    defense_index: int | None
    threat_level: float
    has_edits: bool
    deployed_value: int | None
    iso_x: int
    iso_y: int


def iso_project(col: int, row: int, *, origin_x: int, origin_y: int) -> tuple[int, int]:
    """Convert grid (col, row) to isometric screen coordinates."""
    ix = origin_x + (col - row) * (TILE_W // 2)
    iy = origin_y + (col + row) * (TILE_H // 2)
    return ix, iy


def build_grid(
    instance: "GF01Instance",
    *,
    timestep: int,
    pressure_levels: Mapping[int, int] | None = None,
    history_counts: Mapping[int, int] | None = None,
    history_atoms: list[tuple[int, str, int]] | None = None,
    origin_x: int = 0,
    origin_y: int = 0,
) -> list[TileData]:
    """Build the full tile grid from the instance and current observation state."""
    total_t = len(instance.base_trace)
    t_star = instance.t_star
    mode = instance.mode
    window_size = instance.window_size
    input_aps = list(instance.automaton.input_aps)
    seed = instance.seed

    window_start = t_star if mode == "hard" else max(0, t_star - window_size)
    window_end = t_star

    total_cells = GRID_COLS * GRID_ROWS
    obs_pressure = pressure_levels or {}
    obs_edits = history_counts or {}

    ap_map = defense_tile_assignments(input_aps, total_cells, seed)
    ap_to_cell: dict[str, int] = {v: k for k, v in ap_map.items()}

    deployed_aps: dict[str, int] = {}
    if history_atoms:
        for _t, ap_name, val in history_atoms:
            deployed_aps[ap_name] = val

    tiles: list[TileData] = []

    for idx in range(total_cells):
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        t_frac_start = (idx * total_t) / total_cells
        t_frac_end = ((idx + 1) * total_t) / total_cells
        t_start = int(t_frac_start)
        t_end = max(t_start, int(math.ceil(t_frac_end)) - 1)

        tt = tile_type_for_index(idx, total_cells, seed)

        is_obj = _ranges_overlap(t_start, t_end, window_start, window_end)
        is_current = t_start <= timestep <= t_end

        defense_ap: str | None = ap_map.get(idx)
        defense_index: int | None = None
        if defense_ap is not None:
            try:
                defense_index = input_aps.index(defense_ap)
            except ValueError:
                defense_index = 0

        deployed_value: int | None = None
        if defense_ap is not None and defense_ap in deployed_aps:
            deployed_value = deployed_aps[defense_ap]

        threat = 0.0
        for t in range(t_start, min(t_end + 1, total_t)):
            level = obs_pressure.get(t)
            if level is not None:
                threat = max(threat, level / 10.0)

        has_edits = any(obs_edits.get(t, 0) > 0 for t in range(t_start, t_end + 1))

        if is_obj:
            base = tile_color("objective")
        elif is_current:
            base = tile_color("highlight")
        else:
            base = tile_color(tt)

        if threat > 0.3:
            r, g, b = base
            blend = min(1.0, threat)
            base = (
                int(r + (160 - r) * blend * 0.4),
                int(g * (1.0 - blend * 0.3)),
                int(b * (1.0 - blend * 0.2)),
            )

        ix, iy = iso_project(col, row, origin_x=origin_x, origin_y=origin_y)
        tiles.append(TileData(
            row=row,
            col=col,
            tile_type=tt,
            base_color=base,
            is_objective=is_obj,
            is_current_wave=is_current,
            defense_ap=defense_ap,
            defense_index=defense_index,
            threat_level=threat,
            has_edits=has_edits,
            deployed_value=deployed_value,
            iso_x=ix,
            iso_y=iy,
        ))

    return tiles


def grid_bounding_box(
    tiles: list[TileData],
) -> tuple[int, int, int, int]:
    if not tiles:
        return (0, 0, 0, 0)
    xs = [t.iso_x for t in tiles]
    ys = [t.iso_y for t in tiles]
    return (min(xs), min(ys), max(xs) + TILE_W, max(ys) + TILE_H)


def tile_at_screen_pos(
    tiles: list[TileData], mx: int, my: int
) -> TileData | None:
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
        entries.append({
            "wave": t + 1,
            "t": t,
            "is_current": t == current,
            "is_critical": t == t_star,
            "in_window": window_start <= t <= t_star,
            "pressure": obs_pressure.get(t),
            "has_edits": obs_edits.get(t, 0) > 0,
            "is_past": t < current,
        })
    return entries


def defense_tile_assignments(
    input_aps: list[str],
    total_cells: int,
    seed: int,
) -> dict[int, str]:
    """Spread input APs across grid cells deterministically."""
    assignments: dict[int, str] = {}
    n = max(1, total_cells)
    for i, ap in enumerate(input_aps):
        cell_idx = (i * 7 + seed) % n
        while cell_idx in assignments and cell_idx < n:
            cell_idx = (cell_idx + 1) % n
        assignments[cell_idx] = ap
    return assignments


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or a_start > b_end)
