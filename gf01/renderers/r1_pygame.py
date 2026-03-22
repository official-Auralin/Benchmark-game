"""Causal-board pygame renderer for canonical GF-01-R1 human play.

Presents the benchmark as a spatial control/signal board derived from formal
AP relations rather than arbitrary theme fiction. The renderer consumes only
canonical observation plus static mission metadata, preserving parity with the
text visual path and the JSON renderer.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..semantics import history_counts_by_t, iter_history_atoms
from .r1_theme import (
    COLOR_ACCENT_C,
    COLOR_ACCENT_G,
    COLOR_ACCENT_R,
    COLOR_ACCENT_Y,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_BORDER_HI,
    COLOR_CMD_PTS,
    COLOR_DEPLOY_BTN,
    COLOR_ENERGY,
    COLOR_FLASH_G,
    COLOR_FLASH_R,
    COLOR_HISTORY_OFF,
    COLOR_HISTORY_ON,
    COLOR_PANEL,
    COLOR_PANEL_LITE,
    COLOR_SKIP_BTN,
    COLOR_STAGED_OFF,
    COLOR_STAGED_ON,
    COLOR_TEXT,
    COLOR_TEXT_BRIGHT,
    COLOR_TEXT_DIM,
    COLOR_UNDO_BTN,
    critical_wave_label,
    cmd_points_label,
    defense_color,
    defense_name,
    effect_status_display,
    objective_text_themed,
    threat_name,
    victory_text,
    wave_label,
)
from .r1_grid import (
    GRID_COLS,
    GRID_ROWS,
    TILE_H,
    TILE_W,
    TileData,
    build_grid,
    grid_dimensions,
    tile_at_screen_pos,
    wave_timeline_data,
)

if TYPE_CHECKING:
    from ..models import GF01Instance

_SESSION: "_R1Session | None" = None

SCREEN_W = 1280
SCREEN_H = 800

TOP_BAR_H = 56
SIDEBAR_W = 290
SIDEBAR_X = SCREEN_W - SIDEBAR_W
BOTTOM_H = 150
MAP_X = 16
MAP_Y = TOP_BAR_H + 12
MAP_W = SIDEBAR_X - 32
MAP_H = SCREEN_H - TOP_BAR_H - BOTTOM_H - 24

CARD_W = 148
CARD_H = 105
CARD_GAP = 10
CARD_Y = SCREEN_H - BOTTOM_H + 10
CARD_AREA_X = 16

BTN_W = 138
BTN_H = 38
BTN_GAP = 8
BTN_AREA_X = SIDEBAR_X + 10

TIMELINE_CELL = 30
TIMELINE_GAP = 3
TIMELINE_Y_OFFSET = 420

PRESSURE_BANDS = 10
PRESSURE_MAX_HISTORY = 256
ISO_DEPTH = 14
SPRITE_SIZE = 36
CANONICAL_OBS_KEYS = (
    "t",
    "y_t",
    "effect_status_t",
    "budget_t_remaining",
    "history_atoms",
    "mode",
    "t_star",
    "budget_a_remaining",
    "submitted_action_t",
    "committed_action_t",
)


@dataclass(frozen=True)
class _HitRect:
    x: int
    y: int
    w: int
    h: int
    tag: str
    payload: str

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass
class _DeploymentReport:
    wave: int = -1
    actions: list[tuple[str, int]] = field(default_factory=list)
    deltas: list[tuple[str, str, str]] = field(default_factory=list)

    def clear(self) -> None:
        self.wave = -1
        self.actions.clear()
        self.deltas.clear()


class _PressureHistory:
    def __init__(self) -> None:
        self._levels: OrderedDict[int, int] = OrderedDict()

    def reset(self) -> None:
        self._levels = OrderedDict()

    def record(self, timestep: int, y_t: dict[str, int]) -> None:
        if not y_t:
            return
        on = sum(1 for v in y_t.values() if v == 1)
        ratio = on / max(1, len(y_t))
        level = max(0, min(PRESSURE_BANDS, int(round(ratio * PRESSURE_BANDS))))
        t = int(timestep)
        self._levels.pop(t, None)
        self._levels[t] = level
        while len(self._levels) > PRESSURE_MAX_HISTORY:
            self._levels.popitem(last=False)

    def levels(self) -> Mapping[int, int]:
        return MappingProxyType(self._levels)


def _normalize_binary_map(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in payload.items():
        try:
            bit = int(v)
        except (TypeError, ValueError):
            continue
        if bit in (0, 1):
            out[str(k)] = bit
    return out


def _canonical_inspector_payload(
    *,
    last_obs: dict[str, object] | None,
    instance: "GF01Instance",
) -> dict[str, object]:
    mission = {
        "effect_ap": instance.effect_ap,
        "mode": instance.mode,
        "t_star": int(instance.t_star),
        "budget_timestep": int(instance.budget_timestep),
        "content_hash": instance.content_hash(),
    }
    if instance.budget_atoms is not None:
        mission["budget_atoms"] = int(instance.budget_atoms)
    if not isinstance(last_obs, dict):
        return {"mission": mission, "observation": None}
    observation = {key: last_obs[key] for key in CANONICAL_OBS_KEYS if key in last_obs}
    return {"mission": mission, "observation": observation or None}


def _wrap_json_lines(prefix: str, payload: object, *, width: int = 54) -> list[str]:
    text = json.dumps(payload, sort_keys=True)
    if len(text) <= width:
        return [f"{prefix}{text}"]
    lines = [f"{prefix}{text[:width]}"]
    rest = text[width:]
    while rest:
        lines.append(rest[:width])
        rest = rest[width:]
    return lines


def _inspector_lines(payload: dict[str, object]) -> list[str]:
    lines = ["CANONICAL OBSERVATION"]
    lines.extend(_wrap_json_lines("mission: ", payload.get("mission", {})))
    observation = payload.get("observation")
    if observation is None:
        lines.append("observation: (none yet)")
        return lines
    lines.extend(_wrap_json_lines("observation: ", observation))
    return lines


class _R1Session:
    def __init__(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame backend requested but pygame is not installed; "
                "install with: pip install pygame-ce"
            ) from exc
        self.pg = pygame
        self.pg.init()
        try:
            self.screen = self.pg.display.set_mode((SCREEN_W, SCREEN_H))
        except Exception as exc:
            raise RuntimeError(
                "pygame backend could not open a display; "
                "use --visual-backend text in headless environments"
            ) from exc
        self.pg.display.set_caption("GF-01-R1  -  Causal Board")
        self.clock = self.pg.time.Clock()
        self.font = self.pg.font.SysFont("Arial,Helvetica", 17)
        self.font_sm = self.pg.font.SysFont("Arial,Helvetica", 14)
        self.font_lg = self.pg.font.SysFont("Arial,Helvetica", 24, bold=True)
        self.font_badge = self.pg.font.SysFont("Arial,Helvetica", 14, bold=True)
        self._pressure = _PressureHistory()
        self._prev_y_t: dict[str, int] | None = None
        self._prev_history_atoms_count: int = 0
        self._show_help = True
        self._show_inspector = False
        self._report = _DeploymentReport()
        self._last_committed_action: dict[str, int] | None = None
        self._flash_start: float = 0.0
        self._tile_flashes: dict[str, str] = {}
        self._hovered_card_ap: str | None = None

    # -------------------------------------------------------- procedural icons
    def _draw_defense_sprite(
        self, x: int, y: int, ap_idx: int, color: tuple[int, int, int],
        size: int = SPRITE_SIZE,
    ) -> None:
        cx = x + size // 2
        cy = y + size // 2
        s = size
        variant = ap_idx % 8
        if variant == 0:  # shield dome
            base_y = cy + s // 4
            self.pg.draw.ellipse(self.screen, color,
                                  (x + 2, cy - s // 3, s - 4, s // 2 + 4), 0)
            darker = tuple(max(0, c - 40) for c in color)
            self.pg.draw.ellipse(self.screen, darker,
                                  (x + 2, cy - s // 3, s - 4, s // 2 + 4), 2)
            self.pg.draw.line(self.screen, darker, (x + 3, base_y), (x + s - 3, base_y), 2)
            shine = tuple(min(255, c + 80) for c in color)
            self.pg.draw.arc(self.screen, shine,
                              (x + s // 4, cy - s // 4, s // 3, s // 4),
                              0.3, 2.0, 1)
        elif variant == 1:  # cannon turret
            mount_h = s // 3
            self.pg.draw.rect(self.screen, color,
                               (x + s // 4, cy, s // 2, mount_h))
            darker = tuple(max(0, c - 30) for c in color)
            self.pg.draw.rect(self.screen, darker,
                               (x + s // 4, cy, s // 2, mount_h), 1)
            barrel_y = cy - 2
            self.pg.draw.line(self.screen, color,
                               (cx, barrel_y), (x + s + 2, barrel_y - s // 4), 3)
            self.pg.draw.circle(self.screen, color, (x + s + 2, barrel_y - s // 4), 3)
        elif variant == 2:  # barrier wall (brick pattern)
            bh = max(3, s // 4)
            for row in range(3):
                by = y + row * bh + 2
                offset = (row % 2) * (s // 4)
                for col_i in range(3):
                    bx = x + offset + col_i * (s // 3)
                    bw = s // 3 - 2
                    if bx + bw > x + s:
                        bw = x + s - bx - 1
                    if bw > 0:
                        shade = tuple(max(0, c - row * 12) for c in color)
                        self.pg.draw.rect(self.screen, shade, (bx, by, bw, bh - 1))
                        darker = tuple(max(0, c - 40) for c in color)
                        self.pg.draw.rect(self.screen, darker, (bx, by, bw, bh - 1), 1)
        elif variant == 3:  # crystal emitter
            pts = [(cx, y + 1), (x + s - 3, cy), (cx, y + s - 1), (x + 3, cy)]
            self.pg.draw.polygon(self.screen, color, pts)
            darker = tuple(max(0, c - 40) for c in color)
            self.pg.draw.polygon(self.screen, darker, pts, 2)
            self.pg.draw.line(self.screen, darker, (cx, y + 1), (cx, y + s - 1), 1)
            self.pg.draw.line(self.screen, darker, (x + 3, cy), (x + s - 3, cy), 1)
            shine = tuple(min(255, c + 60) for c in color)
            self.pg.draw.line(self.screen, shine, (cx - 2, y + 4), (x + 5, cy - 2), 1)
        elif variant == 4:  # phase gate
            self.pg.draw.rect(self.screen, color, (x + 2, y + 2, s - 4, s - 4), 2)
            inner = tuple(min(255, c + 30) for c in color)
            self.pg.draw.rect(self.screen, inner, (x + 6, y + 6, s - 12, s - 12), 1)
            self.pg.draw.line(self.screen, color, (cx, y + 2), (cx, y + s - 2), 1)
            self.pg.draw.line(self.screen, color, (x + 2, cy), (x + s - 2, cy), 1)
        elif variant == 5:  # arc pylon
            pts = [(cx, y + 1), (x + s - 3, y + s - 2), (x + 3, y + s - 2)]
            self.pg.draw.polygon(self.screen, color, pts)
            darker = tuple(max(0, c - 40) for c in color)
            self.pg.draw.polygon(self.screen, darker, pts, 2)
            self.pg.draw.line(self.screen, darker, (cx, y + 4), (cx, y + s - 4), 1)
            shine = tuple(min(255, c + 60) for c in color)
            self.pg.draw.circle(self.screen, shine, (cx, y + s // 3), 2)
        elif variant == 6:  # cryo field
            self.pg.draw.circle(self.screen, color, (cx, cy), s // 2, 2)
            inner = tuple(min(255, c + 40) for c in color)
            self.pg.draw.circle(self.screen, inner, (cx, cy), s // 4)
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                ex = cx + int(math.cos(rad) * s * 0.4)
                ey = cy + int(math.sin(rad) * s * 0.4)
                self.pg.draw.line(self.screen, color, (cx, cy), (ex, ey), 1)
        else:  # grav anchor
            self.pg.draw.line(self.screen, color, (cx, y + 2), (cx, y + s - 2), 3)
            self.pg.draw.circle(self.screen, color, (cx, y + s - 5), 4, 2)
            self.pg.draw.line(self.screen, color, (x + 4, cy - 2), (x + s - 4, cy - 2), 2)
            darker = tuple(max(0, c - 30) for c in color)
            self.pg.draw.circle(self.screen, darker, (cx, y + 4), 3)

    # ------------------------------------------------------------------ draw
    def _text(
        self, txt: str, x: int, y: int, *,
        color: tuple[int, int, int] = COLOR_TEXT,
        font: str = "md",
    ) -> None:
        f = self.font_lg if font == "lg" else self.font_sm if font == "sm" else self.font
        if font == "badge":
            f = self.font_badge
        self.screen.blit(f.render(txt, True, color), (x, y))

    def _rect(
        self, x: int, y: int, w: int, h: int, *,
        fill: tuple[int, int, int],
        border: tuple[int, int, int] = COLOR_BORDER,
        bw: int = 1,
        radius: int = 0,
    ) -> None:
        r = self.pg.Rect(x, y, w, h)
        self.pg.draw.rect(self.screen, fill, r, border_radius=radius)
        if bw > 0:
            self.pg.draw.rect(self.screen, border, r, bw, border_radius=radius)

    def _bar(
        self, x: int, y: int, w: int, h: int, *,
        ratio: float,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int] = (32, 38, 52),
    ) -> None:
        self._rect(x, y, w, h, fill=bg, bw=0)
        fill_w = max(0, min(w, int(w * ratio)))
        if fill_w > 0:
            self._rect(x, y, fill_w, h, fill=fg, bw=0)
        self.pg.draw.rect(self.screen, COLOR_BORDER, self.pg.Rect(x, y, w, h), 1)

    # ------------------------------------------------- segmented energy bar
    def _draw_energy_bar(self, x: int, y: int, total: int, remaining: int) -> None:
        seg_w = 18
        seg_h = 14
        gap = 3
        for i in range(total):
            sx = x + i * (seg_w + gap)
            if i < remaining:
                self._rect(sx, y, seg_w, seg_h, fill=COLOR_ENERGY,
                            border=(56, 148, 96), bw=1, radius=2)
            else:
                self._rect(sx, y, seg_w, seg_h, fill=(32, 42, 56),
                            border=COLOR_BORDER, bw=1, radius=2)

    # ------------------------------------------------- iso tile
    def _draw_iso_tile(
        self, tile: TileData, *,
        highlight: bool = False,
        link_highlight: bool = False,
        staged_value: int | None = None,
    ) -> None:
        cx = tile.iso_x + TILE_W // 2
        cy = tile.iso_y + TILE_H // 2
        top_pts = [
            (cx, cy - TILE_H // 2),
            (cx + TILE_W // 2, cy),
            (cx, cy + TILE_H // 2),
            (cx - TILE_W // 2, cy),
        ]
        color = tile.base_color
        if highlight:
            color = tuple(min(255, c + 40) for c in color)
        if link_highlight:
            color = tuple(min(255, c + 25) for c in color)

        flash_key = f"{tile.row},{tile.col}"
        elapsed = time.time() - self._flash_start
        if flash_key in self._tile_flashes and elapsed < 1.5:
            flash_type = self._tile_flashes[flash_key]
            intensity = max(0.0, 1.0 - elapsed / 1.5)
            fc = COLOR_FLASH_G if flash_type == "clear" else COLOR_FLASH_R
            color = tuple(int(c + (fc[i] - c) * intensity * 0.6)
                          for i, c in enumerate(color))
            color = tuple(max(0, min(255, c)) for c in color)

        depth = ISO_DEPTH
        left_pts = [
            (cx - TILE_W // 2, cy),
            (cx, cy + TILE_H // 2),
            (cx, cy + TILE_H // 2 + depth),
            (cx - TILE_W // 2, cy + depth),
        ]
        right_pts = [
            (cx + TILE_W // 2, cy),
            (cx, cy + TILE_H // 2),
            (cx, cy + TILE_H // 2 + depth),
            (cx + TILE_W // 2, cy + depth),
        ]
        left_col = tuple(max(0, c - 30) for c in color)
        right_col = tuple(max(0, c - 50) for c in color)
        self.pg.draw.polygon(self.screen, left_col, left_pts)
        self.pg.draw.polygon(self.screen, right_col, right_pts)
        self.pg.draw.polygon(self.screen, color, top_pts)

        border = COLOR_BORDER_HI if tile.is_objective else COLOR_BORDER
        if tile.is_current_wave:
            border = COLOR_ACCENT_C
        if link_highlight:
            border = COLOR_ACCENT_C
        bw = 2 if (tile.is_objective or tile.is_current_wave or link_highlight) else 1
        self.pg.draw.polygon(self.screen, border, top_pts, bw)

        if staged_value is not None:
            pulse = abs(math.sin(time.time() * 4)) * 0.6 + 0.4
            sc = COLOR_STAGED_ON if staged_value == 1 else COLOR_STAGED_OFF
            sc_alpha = tuple(min(255, int(c * pulse)) for c in sc)
            self.pg.draw.polygon(self.screen, sc_alpha, top_pts, 4)

    # ------------------------------------------------- terrain features
    def _draw_terrain_feature(self, tile: TileData, seed: int) -> None:
        cx = tile.iso_x + TILE_W // 2
        cy = tile.iso_y + TILE_H // 2
        h = ((tile.row * 31 + tile.col * 17 + seed) * 2654435761) & 0xFFFFFFFF
        variant = h % 100
        sc = TILE_W / 96.0

        if tile.tile_type == "rock" and variant < 70:
            bx = cx - int(10 * sc) + (h % 5)
            by = cy - int(6 * sc)
            s = sc
            pts = [
                (bx, by + int(10 * s)), (bx + int(6 * s), by),
                (bx + int(16 * s), by + int(3 * s)),
                (bx + int(18 * s), by + int(11 * s)),
                (bx + int(12 * s), by + int(14 * s)),
                (bx + int(3 * s), by + int(12 * s)),
            ]
            self.pg.draw.polygon(self.screen, (78, 72, 62), pts)
            self.pg.draw.polygon(self.screen, (58, 52, 44), pts, 2)
            if variant < 35:
                ox = int(20 * s)
                pts2 = [
                    (bx + ox, by + int(4 * s)),
                    (bx + ox + int(5 * s), by - int(2 * s)),
                    (bx + ox + int(10 * s), by + int(2 * s)),
                    (bx + ox + int(9 * s), by + int(8 * s)),
                    (bx + ox + int(4 * s), by + int(9 * s)),
                ]
                self.pg.draw.polygon(self.screen, (86, 78, 66), pts2)
                self.pg.draw.polygon(self.screen, (58, 52, 44), pts2, 2)

        elif tile.tile_type == "water":
            for i in range(4):
                wy = cy - int(8 * sc) + i * int(6 * sc)
                wx = cx - int(16 * sc) + (h + i * 7) % int(12 * sc)
                lighter = (58, 108, 168)
                wave_len = int(20 * sc) + i * int(4 * sc)
                self.pg.draw.line(self.screen, lighter,
                                   (wx, wy), (wx + wave_len, wy), 1)
                lighter2 = (48, 92, 148)
                self.pg.draw.line(self.screen, lighter2,
                                   (wx + 2, wy + 2), (wx + wave_len - 4, wy + 2), 1)

        elif tile.tile_type == "ground" and variant > 70:
            mx = cx - int(4 * sc) + (h % int(6 * sc))
            my = cy - TILE_H // 2 - int(4 * sc)
            mw = int(18 * sc) + (h % int(8 * sc))
            mh = int(14 * sc) + (h % int(8 * sc))
            dark = (48, 56, 42)
            mid = (62, 68, 52)
            snow = (192, 200, 210)
            pts_base = [(mx, my + mh), (mx + mw // 2, my), (mx + mw, my + mh)]
            self.pg.draw.polygon(self.screen, dark, pts_base)
            self.pg.draw.polygon(self.screen, mid, pts_base, 2)
            peak_h = mh // 3
            pts_snow = [
                (mx + mw // 4, my + peak_h),
                (mx + mw // 2, my),
                (mx + mw * 3 // 4, my + peak_h),
            ]
            self.pg.draw.polygon(self.screen, snow, pts_snow)
            lighter_snow = (218, 224, 232)
            self.pg.draw.polygon(self.screen, lighter_snow, pts_snow, 1)

    # -------------------------------------------------------- top bar
    def _draw_top_bar(
        self, *, timestep: int, total_waves: int,
        t_star: int, energy: int, energy_total: int,
        cmd_pts: int | None, objective: str, effect_status: str,
    ) -> None:
        self._rect(0, 0, SCREEN_W, TOP_BAR_H, fill=COLOR_PANEL, border=COLOR_BORDER, bw=1)
        self._text(wave_label(timestep, total_waves), 16, 6, font="lg")

        self._text("STEP BUDGET", 220, 6, font="sm", color=COLOR_TEXT_DIM)
        self._draw_energy_bar(220, 24, energy_total, energy)

        if cmd_pts is not None:
            self._text(cmd_points_label(cmd_pts), 440, 10, color=COLOR_CMD_PTS, font="md")

        crit = critical_wave_label(t_star)
        self._text(crit, SCREEN_W - 310, 6, color=COLOR_ACCENT_Y, font="lg")

        status_txt, status_col = effect_status_display(effect_status)
        self._rect(440, 34, len(status_txt) * 8 + 16, 18, fill=status_col, bw=0, radius=3)
        self._text(status_txt, 448, 35, font="sm", color=COLOR_TEXT_BRIGHT)

        self._text(objective, 16, 36, font="md", color=COLOR_ACCENT_Y)

    # -------------------------------------------------------- sidebar
    def _draw_sidebar(
        self, *,
        instance: "GF01Instance",
        timestep: int,
        pending: dict[str, int],
        y_t: dict[str, int],
        pressure_levels: Mapping[int, int],
        history_counts: Mapping[int, int],
        report: _DeploymentReport,
    ) -> None:
        _ = pressure_levels
        sx = SIDEBAR_X
        panel_h = SCREEN_H - TOP_BAR_H - BOTTOM_H
        self._rect(sx, TOP_BAR_H, SIDEBAR_W, panel_h,
                    fill=COLOR_PANEL, border=COLOR_BORDER)

        self._text("SIGNAL STATE", sx + 14, TOP_BAR_H + 10, font="md", color=COLOR_TEXT_BRIGHT)
        y_off = TOP_BAR_H + 34
        output_aps = instance.automaton.output_aps
        for output_ap in output_aps[:8]:
            label = threat_name(output_ap, output_aps)
            value = int(y_t.get(output_ap, 0))
            color = COLOR_ACCENT_G if value == 1 else COLOR_TEXT_DIM
            token = "1" if value == 1 else "0"
            self._text(f"{label}: {token}", sx + 16, y_off, font="sm", color=color)
            y_off += 16

        y_off += 10
        self._text("PENDING CONTROLS", sx + 14, y_off, font="md", color=COLOR_TEXT_BRIGHT)
        y_off += 22
        if pending:
            for ap_name, value in sorted(pending.items())[:6]:
                label = defense_name(ap_name, instance.automaton.input_aps)
                color = COLOR_ACCENT_G if value == 1 else COLOR_ACCENT_R
                self._text(f"{label}: {value}", sx + 16, y_off, font="sm", color=color)
                y_off += 16
        else:
            self._text("(none)", sx + 16, y_off, font="sm", color=COLOR_TEXT_DIM)
            y_off += 16

        if report.wave >= 0:
            y_off += 10
            self._text("LAST COMMITTED STEP", sx + 14, y_off, font="md", color=COLOR_TEXT_BRIGHT)
            y_off += 22
            action_parts: list[str] = []
            input_aps = instance.automaton.input_aps
            for ap_name, val in report.actions[:4]:
                dname = defense_name(ap_name, input_aps)
                tag = "ON" if val == 1 else "OFF"
                action_parts.append(f"{dname} {tag}")
            if action_parts:
                self._text(f"Wave {report.wave + 1}: " + ", ".join(action_parts),
                            sx + 14, y_off, font="sm", color=COLOR_TEXT)
            else:
                self._text(f"Wave {report.wave + 1}: skipped", sx + 14, y_off,
                            font="sm", color=COLOR_TEXT_DIM)
            y_off += 18

            if report.deltas:
                self._text("OBSERVED OUTPUT DELTAS", sx + 14, y_off, font="sm", color=COLOR_TEXT_BRIGHT)
                y_off += 16
                for tname, old_s, new_s in report.deltas[:4]:
                    if old_s != new_s:
                        arrow_col = COLOR_ACCENT_G if new_s == "CLEAR" else COLOR_ACCENT_R
                        self._text(f"{tname}: {old_s} \u2192 {new_s}",
                                    sx + 20, y_off, font="sm", color=arrow_col)
                        y_off += 16
            y_off += 12

        # wave timeline strip
        self._text("STEP TIMELINE", sx + 14, TIMELINE_Y_OFFSET, font="md", color=COLOR_TEXT_BRIGHT)
        total_waves = len(instance.base_trace)
        entries = wave_timeline_data(
            total_waves, timestep, instance.t_star, instance.mode,
            instance.window_size, pressure_levels, history_counts,
        )
        ty = TIMELINE_Y_OFFSET + 24
        cols_per_row = max(1, (SIDEBAR_W - 28) // (TIMELINE_CELL + TIMELINE_GAP))
        for i, entry in enumerate(entries):
            tx = sx + 14 + (i % cols_per_row) * (TIMELINE_CELL + TIMELINE_GAP)
            row_y = ty + (i // cols_per_row) * (TIMELINE_CELL + TIMELINE_GAP)
            fill = COLOR_PANEL_LITE
            border = COLOR_BORDER
            bw = 1
            if entry["in_window"]:
                fill = (58, 52, 34)
            if entry["is_past"]:
                fill = (28, 32, 42)
            if entry["is_current"]:
                fill = (38, 68, 88)
                border = COLOR_ACCENT_C
                bw = 2
            if entry["is_critical"]:
                border = COLOR_ACCENT_Y
                bw = 2
            if entry["has_edits"]:
                fill = tuple(min(255, c + 16) for c in fill)
            self._rect(tx, row_y, TIMELINE_CELL, TIMELINE_CELL,
                        fill=fill, border=border, bw=bw, radius=2)
            label = str(entry["wave"])
            self._text(label, tx + (TIMELINE_CELL - len(label) * 7) // 2,
                        row_y + 7, font="sm")

    # -------------------------------------------------------- iso map
    def _draw_map(
        self, tiles: list[TileData], *,
        instance: "GF01Instance",
        pending: dict[str, int],
        current_outputs: dict[str, int],
        mouse_pos: tuple[int, int],
        hovered_tile: TileData | None,
        hovered_card_ap: str | None,
    ) -> None:
        self._rect(MAP_X - 4, MAP_Y - 4, MAP_W + 8, MAP_H + 8,
                    fill=COLOR_BG, border=COLOR_BORDER, bw=1)

        input_aps = instance.automaton.input_aps
        obj_xs: list[int] = []
        obj_ys: list[int] = []

        for tile in tiles:
            is_hover = (hovered_tile is not None
                        and tile.row == hovered_tile.row
                        and tile.col == hovered_tile.col)
            is_link = (hovered_card_ap is not None
                       and tile.defense_ap == hovered_card_ap)
            staged = pending.get(tile.defense_ap) if tile.defense_ap else None

            self._draw_iso_tile(tile, highlight=is_hover, link_highlight=is_link,
                                 staged_value=staged)

            if tile.is_objective:
                obj_xs.append(tile.iso_x + TILE_W // 2)
                obj_ys.append(tile.iso_y + TILE_H // 2)
                pulse = int(abs((time.time() * 3) % 2.0 - 1.0) * 5) + 3
                ocx = tile.iso_x + TILE_W // 2
                ocy = tile.iso_y + TILE_H // 2
                self.pg.draw.circle(self.screen, COLOR_ACCENT_Y, (ocx, ocy), pulse, 1)

            if tile.defense_ap is not None:
                val = pending.get(tile.defense_ap)
                if val == 1:
                    col = defense_color(tile.defense_ap, input_aps)
                elif val == 0:
                    col = COLOR_TEXT_DIM
                else:
                    col = (100, 108, 124)
                ap_idx = tile.defense_index if tile.defense_index is not None else 0
                self._draw_defense_sprite(
                    tile.iso_x + TILE_W // 2 - SPRITE_SIZE // 2,
                    tile.iso_y + TILE_H // 2 - SPRITE_SIZE // 2 - 4,
                    ap_idx, col, size=SPRITE_SIZE,
                )
                label = defense_name(tile.defense_ap, input_aps)
                self._text(
                    label[:10],
                    tile.iso_x + 12,
                    tile.iso_y + TILE_H // 2 + 10,
                    font="sm",
                    color=COLOR_TEXT_BRIGHT,
                )

            if tile.output_ap is not None:
                value = int(current_outputs.get(tile.output_ap, 0))
                col = COLOR_ACCENT_G if value == 1 else COLOR_ACCENT_R
                cx = tile.iso_x + TILE_W // 2
                cy = tile.iso_y + TILE_H // 2 - 2
                self.pg.draw.circle(self.screen, col, (cx, cy), 14)
                self.pg.draw.circle(self.screen, COLOR_BORDER_HI, (cx, cy), 14, 2)
                self._text(
                    threat_name(tile.output_ap, instance.automaton.output_aps)[:10],
                    tile.iso_x + 12,
                    tile.iso_y + TILE_H // 2 + 10,
                    font="sm",
                    color=COLOR_TEXT_BRIGHT,
                )

            if tile.deployed_value is not None and tile.defense_ap not in pending:
                dot_col = COLOR_HISTORY_ON if tile.deployed_value == 1 else COLOR_HISTORY_OFF
                dx = tile.iso_x + TILE_W // 2 - SPRITE_SIZE // 2 - 6
                dy = tile.iso_y + TILE_H // 2 + 8
                self.pg.draw.circle(self.screen, dot_col, (dx, dy), 6)
                self.pg.draw.circle(self.screen, (18, 22, 32), (dx, dy), 6, 1)
                marker = "\u25B2" if tile.deployed_value == 1 else "\u25BC"
                self._text(marker, dx - 4, dy - 6, font="badge", color=COLOR_TEXT_BRIGHT)

        if obj_xs:
            obj_label = objective_text_themed(instance)
            label_w = len(obj_label) * 9 + 16
            label_x = sum(obj_xs) // len(obj_xs) - label_w // 2
            label_y = min(obj_ys) - 32
            bg_surf = self.pg.Surface((label_w, 26), self.pg.SRCALPHA)
            bg_surf.fill((14, 18, 26, 210))
            self.screen.blit(bg_surf, (label_x - 4, label_y - 3))
            self.pg.draw.rect(self.screen, COLOR_ACCENT_Y,
                               (label_x - 4, label_y - 3, label_w, 26), 2)
            self._text(obj_label, label_x + 4, label_y + 2, font="sm", color=COLOR_ACCENT_Y)

        if hovered_tile is not None:
            tooltip_x = mouse_pos[0] + 16
            tooltip_y = mouse_pos[1] - 28
            if hovered_tile.defense_ap:
                name = defense_name(hovered_tile.defense_ap, input_aps)
                label = name
                bg_surf = self.pg.Surface((len(label) * 8 + 12, 20), self.pg.SRCALPHA)
                bg_surf.fill((14, 18, 26, 180))
                self.screen.blit(bg_surf, (tooltip_x - 4, tooltip_y - 2))
                self._text(label, tooltip_x, tooltip_y, font="sm", color=COLOR_TEXT_BRIGHT)
            elif hovered_tile.output_ap:
                label = threat_name(hovered_tile.output_ap, instance.automaton.output_aps)
                bg_surf = self.pg.Surface((len(label) * 8 + 12, 20), self.pg.SRCALPHA)
                bg_surf.fill((14, 18, 26, 180))
                self.screen.blit(bg_surf, (tooltip_x - 4, tooltip_y - 2))
                self._text(label, tooltip_x, tooltip_y, font="sm", color=COLOR_TEXT_BRIGHT)
            elif hovered_tile.is_objective:
                bg_surf = self.pg.Surface((110, 20), self.pg.SRCALPHA)
                bg_surf.fill((14, 18, 26, 180))
                self.screen.blit(bg_surf, (tooltip_x - 4, tooltip_y - 2))
                self._text("Target Output", tooltip_x, tooltip_y,
                            font="sm", color=COLOR_ACCENT_Y)

    # ------------------------------------------------- defense cards
    def _draw_defense_cards(
        self, *,
        input_aps: list[str],
        pending: dict[str, int],
        page: int,
        page_size: int,
        hovered_card_ap: str | None,
    ) -> tuple[list[_HitRect], int, int, str | None]:
        total = len(input_aps)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(0, page), total_pages - 1)
        start = page * page_size
        visible = input_aps[start:start + page_size]

        hits: list[_HitRect] = []
        new_hover_ap: str | None = None

        self._rect(0, SCREEN_H - BOTTOM_H, SIDEBAR_X, BOTTOM_H,
                    fill=COLOR_PANEL, border=COLOR_BORDER)

        self._text("CONTROL INPUTS", CARD_AREA_X, CARD_Y - 12, font="md", color=COLOR_TEXT_BRIGHT)
        self._text(
            f"Page {page + 1}/{total_pages}  [\u2190/\u2192]  [+/-]",
            CARD_AREA_X + 100, CARD_Y - 12, font="sm", color=COLOR_TEXT_DIM,
        )

        mouse_x, mouse_y = self.pg.mouse.get_pos()

        for idx, ap in enumerate(visible):
            cx = CARD_AREA_X + idx * (CARD_W + CARD_GAP)
            cy = CARD_Y + 4
            if cx + CARD_W > SIDEBAR_X - 8:
                break

            val = pending.get(ap)
            name = defense_name(ap, input_aps)
            d_col = defense_color(ap, input_aps)
            try:
                ap_idx = input_aps.index(ap)
            except ValueError:
                ap_idx = 0

            card_rect = self.pg.Rect(cx, cy, CARD_W, CARD_H)
            is_hovered = card_rect.collidepoint(mouse_x, mouse_y)
            if is_hovered:
                new_hover_ap = ap

            if val == 1:
                fill = (34, 58, 48)
                border = d_col
            elif val == 0:
                fill = (48, 36, 34)
                border = COLOR_ACCENT_R
            else:
                fill = COLOR_PANEL_LITE
                border = COLOR_BORDER
            if is_hovered:
                fill = tuple(min(255, c + 15) for c in fill)
                border = COLOR_ACCENT_C

            self._rect(cx, cy, CARD_W, CARD_H, fill=fill, border=border, bw=2, radius=4)

            badge_num = str(ap_idx + 1)
            self._rect(cx + 4, cy + 4, 22, 20, fill=(16, 22, 36),
                        border=COLOR_ACCENT_C, bw=2, radius=3)
            self._text(badge_num, cx + 8, cy + 5, font="badge", color=COLOR_ACCENT_C)

            self._draw_defense_sprite(cx + 26, cy + 4, ap_idx, d_col, size=22)
            self._text(name, cx + 52, cy + 8, font="sm", color=COLOR_TEXT)

            status_txt = "1" if val == 1 else "0" if val == 0 else "\u2014"
            status_col = COLOR_ACCENT_G if val == 1 else COLOR_ACCENT_R if val == 0 else COLOR_TEXT_DIM
            self._text(status_txt, cx + 52, cy + 28, font="sm", color=status_col)

            on_x = cx + 10
            on_y = cy + CARD_H - 34
            on_fill = (38, 72, 56) if val == 1 else COLOR_PANEL_LITE
            on_bdr = COLOR_ACCENT_G if val == 1 else COLOR_BORDER
            self._rect(on_x, on_y, 56, 26, fill=on_fill, border=on_bdr, bw=1, radius=3)
            self._text("ON", on_x + 18, on_y + 5, font="md",
                        color=COLOR_ACCENT_G if val == 1 else COLOR_TEXT_DIM)
            hits.append(_HitRect(on_x, on_y, 56, 26, "set", ap + ":1"))

            off_x = cx + 76
            off_fill = (72, 38, 34) if val == 0 else COLOR_PANEL_LITE
            off_bdr = COLOR_ACCENT_R if val == 0 else COLOR_BORDER
            self._rect(off_x, on_y, 56, 26, fill=off_fill, border=off_bdr, bw=1, radius=3)
            self._text("OFF", off_x + 14, on_y + 5, font="md",
                        color=COLOR_ACCENT_R if val == 0 else COLOR_TEXT_DIM)
            hits.append(_HitRect(off_x, on_y, 56, 26, "set", ap + ":0"))

        return hits, page, total_pages, new_hover_ap

    # ------------------------------------------------- action buttons (stacked right)
    def _draw_action_buttons(self, pending_count: int) -> list[_HitRect]:
        hits: list[_HitRect] = []
        bx = BTN_AREA_X
        by = SCREEN_H - BOTTOM_H + 10

        deploy_label = f"COMMIT ({pending_count})" if pending_count > 0 else "COMMIT"
        self._rect(bx, by, BTN_W, BTN_H,
                    fill=COLOR_DEPLOY_BTN if pending_count > 0 else (38, 68, 52),
                    border=COLOR_ACCENT_G, bw=2, radius=5)
        self._text(deploy_label, bx + 16, by + 8, font="md", color=COLOR_TEXT_BRIGHT)
        hits.append(_HitRect(bx, by, BTN_W, BTN_H, "action", "deploy"))

        by2 = by + BTN_H + BTN_GAP
        self._rect(bx, by2, BTN_W, BTN_H,
                    fill=COLOR_SKIP_BTN, border=COLOR_BORDER, bw=1, radius=5)
        self._text("SKIP STEP", bx + 22, by2 + 8, font="md", color=COLOR_TEXT_DIM)
        hits.append(_HitRect(bx, by2, BTN_W, BTN_H, "action", "skip"))

        by3 = by2 + BTN_H + BTN_GAP
        self._rect(bx, by3, BTN_W, BTN_H,
                    fill=COLOR_UNDO_BTN if pending_count > 0 else (58, 38, 36),
                    border=COLOR_ACCENT_R if pending_count > 0 else COLOR_BORDER,
                    bw=1, radius=5)
        self._text("CLEAR ALL", bx + 20, by3 + 8, font="md",
                    color=COLOR_TEXT_BRIGHT if pending_count > 0 else COLOR_TEXT_DIM)
        hits.append(_HitRect(bx, by3, BTN_W, BTN_H, "action", "recall"))

        return hits

    # ------------------------------------------------- wave transition
    def _draw_wave_transition(
        self, wave_num: int, report: _DeploymentReport | None = None,
    ) -> None:
        overlay = self.pg.Surface((SCREEN_W, SCREEN_H), self.pg.SRCALPHA)
        overlay.fill((14, 18, 26, 200))
        self.screen.blit(overlay, (0, 0))

        if report and report.wave >= 0 and report.actions:
            self._text(f"STEP {report.wave + 1} COMMITTED",
                        SCREEN_W // 2 - 100, SCREEN_H // 2 - 60,
                        font="lg", color=COLOR_ACCENT_C)
            y_off = SCREEN_H // 2 - 24
            for ap_name, val in report.actions[:6]:
                tag = "SET TO 1" if val == 1 else "SET TO 0"
                col = COLOR_ACCENT_G if val == 1 else COLOR_ACCENT_R
                self._text(f"{ap_name}: {tag}", SCREEN_W // 2 - 80, y_off,
                            font="sm", color=col)
                y_off += 18
            if report.deltas:
                y_off += 6
                for tname, old_s, new_s in report.deltas[:4]:
                    if old_s != new_s:
                        col = COLOR_ACCENT_G if new_s == "CLEAR" else COLOR_ACCENT_R
                        self._text(f"{tname}: {old_s} \u2192 {new_s}",
                                    SCREEN_W // 2 - 80, y_off, font="sm", color=col)
                        y_off += 18
        else:
            txt = f"STEP {wave_num}"
            self._text(txt, SCREEN_W // 2 - len(txt) * 7, SCREEN_H // 2 - 24,
                        font="lg", color=COLOR_ACCENT_C)
            sub = "Prepare your next control assignment"
            self._text(sub, SCREEN_W // 2 - len(sub) * 4, SCREEN_H // 2 + 12,
                        font="sm", color=COLOR_TEXT_DIM)
        self.pg.display.flip()
        self.pg.time.wait(800)

    # ------------------------------------------------- victory / defeat
    def _draw_end_screen(self, *, goal: bool, suff: bool, min1: bool) -> None:
        title, subtitle, color = victory_text(goal, suff, min1)
        overlay = self.pg.Surface((SCREEN_W, SCREEN_H), self.pg.SRCALPHA)
        overlay.fill((14, 18, 26, 220))
        self.screen.blit(overlay, (0, 0))

        bw = 420
        bh = 170
        bx = (SCREEN_W - bw) // 2
        by = (SCREEN_H - bh) // 2
        self._rect(bx, by, bw, bh, fill=COLOR_PANEL, border=color, bw=3, radius=8)
        self._text(title, bx + (bw - len(title) * 14) // 2, by + 30, font="lg", color=color)
        words = subtitle.split()
        line = ""
        ly = by + 76
        for word in words:
            test = (line + " " + word).strip()
            if len(test) * 8 > bw - 40:
                self._text(line, bx + 20, ly, font="sm", color=COLOR_TEXT)
                ly += 20
                line = word
            else:
                line = test
        if line:
            self._text(line, bx + 20, ly, font="sm", color=COLOR_TEXT)
        self._text("Press any key to continue", bx + 90, by + bh - 30,
                    font="sm", color=COLOR_TEXT_DIM)
        self.pg.display.flip()
        waiting = True
        while waiting:
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    return
                if event.type in (self.pg.KEYDOWN, self.pg.MOUSEBUTTONDOWN):
                    waiting = False

    # ------------------------------------------------- help overlay
    def _draw_help_overlay(self) -> None:
        lines = [
            "CAUSAL BOARD  -  QUICK HELP",
            "",
            "Goal: choose control inputs that trigger the target output in time.",
            "Each card stages a 0/1 assignment for one input proposition.",
            "",
            "Mouse: Click ON/OFF on control cards to stage assignments.",
            "       Click an input tile to toggle its staged value.",
            "       Hover a card to highlight its board location.",
            "",
            "Keys:  1-9,0   Toggle visible control cards",
            "       Enter   Commit staged assignments",
            "       Escape  Skip step (no assignment)",
            "       Backsp  Clear all staged assignments",
            "       \u2190/\u2192     Page through controls",
            "       +/-     Adjust cards per page",
            "       I       Toggle canonical observation inspector",
            "       H       Toggle this help panel",
            "",
            "After committing, check LAST COMMITTED STEP in the sidebar",
            "to see what changed. Press H to dismiss.",
        ]
        w = 520
        h = 30 + len(lines) * 22
        x = (SCREEN_W - w) // 2
        y = (SCREEN_H - h) // 2
        self._rect(x, y, w, h, fill=(18, 22, 32), border=COLOR_BORDER_HI, bw=2, radius=6)
        for i, line in enumerate(lines):
            col = COLOR_TEXT_BRIGHT if i == 0 else COLOR_TEXT
            self._text(line, x + 20, y + 16 + i * 22, font="sm", color=col)

    def _draw_inspector_overlay(self, payload: dict[str, object]) -> None:
        lines = _inspector_lines(payload)
        width = 420
        height = 24 + len(lines) * 18
        x = SCREEN_W - width - 18
        y = TOP_BAR_H + 18
        self._rect(
            x,
            y,
            width,
            height,
            fill=(18, 22, 32),
            border=COLOR_BORDER_HI,
            bw=2,
            radius=6,
        )
        for idx, line in enumerate(lines):
            color = COLOR_TEXT_BRIGHT if idx == 0 else COLOR_TEXT
            self._text(line, x + 14, y + 10 + idx * 18, font="sm", color=color)

    # ================================================ main entry point
    def choose_action(
        self, *,
        last_obs: dict[str, object] | None,
        timestep: int,
        instance: "GF01Instance",
        objective_text: str,
    ) -> dict[str, int] | None:
        pending: dict[str, int] = {}
        page = 0
        page_size = 6
        input_aps = list(instance.automaton.input_aps)
        output_aps = list(instance.automaton.output_aps)
        total_waves = len(instance.base_trace)

        if timestep == 0:
            self._prev_y_t = None
            self._prev_history_atoms_count = 0
            self._pressure.reset()
            self._show_help = True
            self._show_inspector = False
            self._report.clear()
            self._tile_flashes.clear()
            self._hovered_card_ap = None
            self._last_committed_action = None
            self._draw_wave_transition(1)

        current_y_t = _normalize_binary_map(
            None if last_obs is None else last_obs.get("y_t", {})
        )
        history_atoms_raw = [] if last_obs is None else last_obs.get("history_atoms", [])
        parsed_atoms = iter_history_atoms(history_atoms_raw)
        changed_outputs: list[str] = []

        if last_obs is not None and self._prev_y_t is not None:
            if len(parsed_atoms) > self._prev_history_atoms_count:
                new_atoms = parsed_atoms[self._prev_history_atoms_count:]
                self._report.wave = max(0, timestep - 1)
                self._report.actions = [(ap, val) for _t, ap, val in new_atoms]
                self._report.deltas = []
                for oap in output_aps:
                    old_val = self._prev_y_t.get(oap, 0)
                    new_val = current_y_t.get(oap, 0)
                    old_s = str(old_val)
                    new_s = str(new_val)
                    tname = threat_name(oap, output_aps)
                    self._report.deltas.append((tname, old_s, new_s))
                    if old_val != new_val:
                        changed_outputs.append(oap)

                self._draw_wave_transition(timestep + 1, self._report)
            elif timestep > 0 and len(parsed_atoms) == self._prev_history_atoms_count:
                self._draw_wave_transition(timestep + 1)

        if last_obs is not None:
            obs_t = max(0, timestep - 1)
            self._pressure.record(obs_t, current_y_t)
        if current_y_t:
            self._prev_y_t = dict(current_y_t)
        self._prev_history_atoms_count = len(parsed_atoms)

        pressure_levels = self._pressure.levels()
        h_counts = history_counts_by_t(history_atoms_raw)

        energy_total = int(instance.budget_timestep)
        energy = (
            int(last_obs["budget_t_remaining"])
            if last_obs and "budget_t_remaining" in last_obs
            else energy_total
        )
        cmd_pts = None
        if last_obs and "budget_a_remaining" in last_obs:
            cmd_pts = int(last_obs["budget_a_remaining"])
        elif instance.budget_atoms is not None:
            cmd_pts = int(instance.budget_atoms)
        effect_status = (
            str(last_obs.get("effect_status_t", "unknown"))
            if last_obs else "unknown"
        )

        grid_cols, grid_rows = grid_dimensions(instance)
        iso_width = (grid_cols + grid_rows - 1) * (TILE_W // 2)
        iso_height = (grid_cols + grid_rows - 1) * (TILE_H // 2) + ISO_DEPTH
        grid_origin_x = MAP_X + (MAP_W - iso_width) // 2 + grid_rows * (TILE_W // 2)
        grid_origin_y = MAP_Y + (MAP_H - iso_height) // 2

        tiles = build_grid(
            instance,
            timestep=timestep,
            pressure_levels=pressure_levels,
            history_counts=h_counts,
            history_atoms=parsed_atoms,
            current_outputs=current_y_t,
            origin_x=grid_origin_x,
            origin_y=grid_origin_y,
        )
        if changed_outputs:
            self._tile_flashes.clear()
            self._flash_start = time.time()
            for tile in tiles:
                if tile.output_ap in changed_outputs:
                    key = f"{tile.row},{tile.col}"
                    new_val = current_y_t.get(tile.output_ap or "", 0)
                    self._tile_flashes[key] = "clear" if new_val == 0 else "threat"

        card_hits: list[_HitRect] = []
        btn_hits: list[_HitRect] = []

        while True:
            self.clock.tick(30)
            mouse_pos = self.pg.mouse.get_pos()
            hovered_tile = tile_at_screen_pos(tiles, mouse_pos[0], mouse_pos[1])

            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:
                    return None
                if event.type == self.pg.KEYDOWN:
                    if event.key in (self.pg.K_RETURN, self.pg.K_KP_ENTER):
                        return dict(pending)
                    if event.key == self.pg.K_ESCAPE:
                        return {}
                    if event.key == self.pg.K_BACKSPACE:
                        pending.clear()
                    if event.key == self.pg.K_h:
                        self._show_help = not self._show_help
                    if event.key == self.pg.K_i:
                        self._show_inspector = not self._show_inspector
                    if event.key in (self.pg.K_LEFT, self.pg.K_PAGEUP):
                        page = max(0, page - 1)
                    if event.key in (self.pg.K_RIGHT, self.pg.K_PAGEDOWN):
                        page += 1
                    if event.key in (self.pg.K_EQUALS, self.pg.K_PLUS, self.pg.K_KP_PLUS):
                        page_size = min(10, page_size + 1)
                    if event.key in (self.pg.K_MINUS, self.pg.K_UNDERSCORE, self.pg.K_KP_MINUS):
                        page_size = max(3, page_size - 1)

                    key_map = {
                        self.pg.K_1: 0, self.pg.K_2: 1, self.pg.K_3: 2,
                        self.pg.K_4: 3, self.pg.K_5: 4, self.pg.K_6: 5,
                        self.pg.K_7: 6, self.pg.K_8: 7, self.pg.K_9: 8,
                        self.pg.K_0: 9,
                    }
                    ki = key_map.get(event.key)
                    if ki is not None:
                        tp = max(1, (len(input_aps) + page_size - 1) // page_size)
                        clamped_page = min(max(0, page), tp - 1)
                        vis_start = clamped_page * page_size
                        vis = input_aps[vis_start:vis_start + page_size]
                        if ki < len(vis):
                            ap = vis[ki]
                            cur = pending.get(ap)
                            if cur is None:
                                pending[ap] = 1
                            elif cur == 1:
                                pending[ap] = 0
                            else:
                                pending.pop(ap, None)

                if event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    consumed = False
                    for hr in card_hits:
                        if not hr.contains(mx, my):
                            continue
                        ap_str, val_str = hr.payload.split(":", 1)
                        v = int(val_str)
                        if pending.get(ap_str) == v:
                            pending.pop(ap_str, None)
                        else:
                            pending[ap_str] = v
                        consumed = True
                        break
                    if not consumed:
                        for hr in btn_hits:
                            if not hr.contains(mx, my):
                                continue
                            if hr.payload == "deploy":
                                return dict(pending)
                            elif hr.payload == "skip":
                                return {}
                            elif hr.payload == "recall":
                                pending.clear()
                            consumed = True
                            break
                    if not consumed and hovered_tile and hovered_tile.defense_ap:
                        ap = hovered_tile.defense_ap
                        cur = pending.get(ap)
                        if cur is None:
                            pending[ap] = 1
                        elif cur == 1:
                            pending[ap] = 0
                        else:
                            pending.pop(ap, None)

            # ---- draw frame ----
            self.screen.fill(COLOR_BG)

            self._draw_top_bar(
                timestep=timestep,
                total_waves=total_waves,
                t_star=instance.t_star,
                energy=energy,
                energy_total=energy_total,
                cmd_pts=cmd_pts,
                objective=objective_text_themed(instance),
                effect_status=effect_status,
            )

            self._draw_map(
                tiles,
                instance=instance,
                pending=pending,
                current_outputs=current_y_t,
                mouse_pos=mouse_pos,
                hovered_tile=hovered_tile,
                hovered_card_ap=self._hovered_card_ap,
            )

            self._draw_sidebar(
                instance=instance,
                timestep=timestep,
                pending=pending,
                y_t=current_y_t,
                pressure_levels=pressure_levels,
                history_counts=h_counts,
                report=self._report,
            )

            card_hits, page, _, self._hovered_card_ap = self._draw_defense_cards(
                input_aps=input_aps,
                pending=pending,
                page=page,
                page_size=page_size,
                hovered_card_ap=self._hovered_card_ap,
            )
            btn_hits = self._draw_action_buttons(len(pending))

            self._text(
                "H=Help  I=Inspector  Enter=Deploy  Esc=Skip  Backspace=Undo  "
                "\u2190/\u2192=Page  1-9=Toggle",
                16, SCREEN_H - 18, font="sm", color=COLOR_TEXT_DIM,
            )

            if self._show_inspector:
                self._draw_inspector_overlay(
                    _canonical_inspector_payload(
                        last_obs=last_obs,
                        instance=instance,
                    )
                )
            if self._show_help:
                self._draw_help_overlay()

            self.pg.display.flip()

def _session() -> _R1Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _R1Session()
    return _SESSION


def choose_action_pygame(
    *,
    last_obs: dict[str, object] | None,
    timestep: int,
    instance: "GF01Instance",
    objective_text: str,
) -> dict[str, int] | None:
    return _session().choose_action(
        last_obs=last_obs,
        timestep=timestep,
        instance=instance,
        objective_text=objective_text,
    )
