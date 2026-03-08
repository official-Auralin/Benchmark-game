"""
Optional pygame backend for map-first GF-01-R1 human interaction.

This backend provides a minimal graphical control surface for one timestep at a
time. It renders an observation-safe timeline/grid view, lets the player set
per-AP values for the current timestep, and returns a machine-checkable action
dictionary compatible with the existing episode loop.
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
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..semantics import history_counts_by_t, timeline_marker_for_t

if TYPE_CHECKING:
    from ..models import GF01Instance


from .r1_pygame_helpers import (
    COMMAND_RESPONSE_LINE_MAX_LEN,
    COMMAND_RESPONSE_MAX_ENTRIES,
    COMMAND_RESPONSE_MAX_VISIBLE,
    SECTOR_BOARD_COLS,
    SECTOR_BOARD_ROWS,
    SECTOR_PRESSURE_BANDS,
    SECTOR_PRESSURE_HISTORY_MAX,
    TIMELINE_MINIMAP_CHARS,
    UI_TEXT_MIN_TRUNCATE_LEN,
    _Button,
    _CommandResponseTrailModel,
    _SectorBoardCell,
    _SectorPressureHistoryModel,
    _WaveStripModel,
    _ap_group_key,
    _apply_group_filter,
    _build_sector_board_cells,
    _build_timeline_minimap,
    _bucket_index_for_t,
    _bucket_marker,
    _bucketize_history_counts,
    _bucketize_pressure_levels,
    _canonical_exposure_payload,
    _clamp_page_size,
    _clamp_timeline_span,
    _command_console_lines,
    _command_row_status,
    _command_console_stage_status,
    _command_console_sector_tokens,
    _control_visible_pool,
    _cycle_pending_bit,
    _describe_output_delta,
    _edits_token,
    _effect_status_badge,
    _format_top_pressure_summary,
    _group_rows_for_controls,
    _grouped_input_aps,
    _help_overlay_lines,
    _normalize_binary_map,
    _objective_window_bounds,
    _objective_window_pressure_summary,
    _observation_inspector_lines,
    _onboarding_strip_lines,
    _pending_loadout_entries,
    _paginate_input_aps,
    _place_minimap_bracket,
    _pressure_level_from_observation,
    _pressure_token,
    _range_contains_t,
    _sector_board_pending_badge_text,
    _sector_board_col_label,
    _ranges_overlap,
    _sector_board_cell_glyph,
    _sector_board_cell_name,
    _sector_board_hud_sections,
    _sector_board_legend_lines,
    _sector_bucket_bounds,
    _sector_pressure_fill,
    _summarize_committed_action,
    _summarize_observed_outputs,
    _summarize_pending_interventions,
    _summarize_visible_ap_groups,
    _timeline_mark,
    _timeline_window_bounds,
    _top_pressure_sectors,
    _truncate_ui_text,
    _wave_pressure_strip_state,
)


_SESSION: "_R1PygameSession | None" = None

SECTOR_BOARD_LEGEND_X_OFFSET = 14
SECTOR_BOARD_LEGEND_Y_OFFSET = 30
SECTOR_BOARD_LEGEND_LINE_STEP = 18
SECTOR_BOARD_HUD_X_OFFSET = 220
SECTOR_BOARD_CARD_ROW_1_Y = 46
SECTOR_BOARD_CARD_ROW_2_Y = 108
SECTOR_BOARD_CARD_W = 144
SECTOR_BOARD_CARD_GAP_X = 10
SECTOR_BOARD_CARD_LINE_X = 10
SECTOR_BOARD_CARD_TITLE_Y = 8
SECTOR_BOARD_CARD_LINE_Y = 24
SECTOR_BOARD_CARD_LINE_STEP = 14
SECTOR_BOARD_CARD_FILL = (28, 38, 54)
SECTOR_BOARD_PENDING_BADGE_SIZE = 14
SECTOR_BOARD_PENDING_BADGE_OFFSET_X = 4
SECTOR_BOARD_PENDING_BADGE_OFFSET_Y = 2
SECTOR_BOARD_HUD_PRIMARY_COLOR = (198, 212, 234)
SECTOR_BOARD_HUD_SECONDARY_COLOR = (176, 191, 216)
SECTOR_BOARD_HUD_CARD_ACCENTS = (
    (168, 114, 76),
    (142, 132, 94),
    (98, 148, 188),
    (126, 196, 134),
)
COMMAND_CONSOLE_X = 16
COMMAND_CONSOLE_Y_OFFSET = 56
COMMAND_CONSOLE_W = 580
COMMAND_CONSOLE_TITLE_Y = 42
COMMAND_CONSOLE_INFO_1_Y = 20
COMMAND_CONSOLE_INFO_2_Y = 2
COMMAND_CONSOLE_LOADOUT_LABEL_Y = 38
COMMAND_CONSOLE_LOADOUT_Y = 56
COMMAND_CONSOLE_PAGE_Y = 84
COMMAND_CONSOLE_GROUPS_Y = 104
COMMAND_CONSOLE_ROWS_Y = 112
COMMAND_CONSOLE_MIN_H = 196
COMMAND_CONSOLE_LOADOUT_CHIP_W = 92
COMMAND_CONSOLE_LOADOUT_CHIP_H = 22
COMMAND_CONSOLE_LOADOUT_CHIP_GAP = 8
COMMAND_CONSOLE_SECTOR_CHIP_W = 98
COMMAND_CONSOLE_SECTOR_CHIP_H = 20
COMMAND_CONSOLE_SECTOR_CHIP_GAP = 8
COMMAND_CONSOLE_SECTOR_CHIP_X = 286
COMMAND_CONSOLE_SECTOR_CHIP_Y = 16
COMMAND_CONSOLE_STAGE_BADGE_X = 186
COMMAND_CONSOLE_STAGE_BADGE_Y = 40
COMMAND_CONSOLE_STAGE_BADGE_W = 120
COMMAND_CONSOLE_STAGE_BADGE_H = 22
COMMAND_CONSOLE_ROW_BADGE_X = 98
COMMAND_CONSOLE_ROW_BADGE_W = 86
COMMAND_CONSOLE_ROW_BADGE_H = 20
COMMAND_CONSOLE_ROW_CARD_W = 376
COMMAND_CONSOLE_ROW_CARD_H = 32

class _R1PygameSession:
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
            self.screen = self.pg.display.set_mode((1200, 760))
        except Exception as exc:  # pragma: no cover - platform/display specific
            raise RuntimeError(
                "pygame backend could not open a display window; "
                "use --visual-backend text in headless environments"
            ) from exc
        self.pg.display.set_caption("GF-01-R1 Map-First Visual")
        self.clock = self.pg.time.Clock()
        self.font = self.pg.font.SysFont("Courier New", 18)
        self.font_small = self.pg.font.SysFont("Courier New", 14)
        self.font_title = self.pg.font.SysFont("Courier New", 24, bold=True)
        self._previous_observed_y_t: dict[str, int] | None = None
        self._last_committed_action_summary: str | None = None
        self._last_committed_t: int | None = None
        self._wave_strip = _WaveStripModel()
        self._sector_pressure_history = _SectorPressureHistoryModel()
        self._command_response_trail = _CommandResponseTrailModel()
        self._command_focus_timesteps: list[int] = []
        self._hovered_sector_range: tuple[int, int] | None = None
        self._sector_board_hitboxes: list[tuple[int, int, int, int, _SectorBoardCell]] = []
        self._loadout_chip_hitboxes: list[tuple[int, int, int, int, str]] = []
        self._pinned_sector_coords: tuple[int, int] | None = None
        self._live_sector_name: str | None = None
        self._target_sector_name: str | None = None
        self._pinned_sector_name: str | None = None
        self._pinned_sector_range: tuple[int, int] | None = None
        self._show_help_overlay = True
        self._show_observation_inspector = False

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        *,
        color: tuple[int, int, int] = (220, 230, 245),
        small: bool = False,
        title: bool = False,
    ) -> None:
        font = self.font_title if title else self.font_small if small else self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _record_command_focus(self, timestep: int) -> None:
        t = int(timestep)
        if self._command_focus_timesteps and self._command_focus_timesteps[0] == t:
            return
        self._command_focus_timesteps.insert(0, t)
        if len(self._command_focus_timesteps) > 3:
            self._command_focus_timesteps = self._command_focus_timesteps[:3]

    def _draw_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        fill: tuple[int, int, int],
        border: tuple[int, int, int] = (70, 85, 110),
        border_width: int = 1,
    ) -> None:
        self.pg.draw.rect(self.screen, fill, self.pg.Rect(x, y, w, h))
        self.pg.draw.rect(self.screen, border, self.pg.Rect(x, y, w, h), border_width)

    def _draw_pressure_band(self, *, x: int, y0: int, cell_w: int, level: int) -> None:
        self._draw_rect(
            x + 2,
            y0 + 2,
            cell_w - 4,
            6,
            fill=_sector_pressure_fill(level),
            border=(88, 103, 128),
            border_width=1,
        )

    def _draw_sector_board_legend(self, *, x: int, y: int) -> None:
        for idx, line in enumerate(_sector_board_legend_lines()):
            self._draw_text(
                line,
                x + SECTOR_BOARD_LEGEND_X_OFFSET,
                y + SECTOR_BOARD_LEGEND_Y_OFFSET + idx * SECTOR_BOARD_LEGEND_LINE_STEP,
                small=True,
                color=SECTOR_BOARD_HUD_SECONDARY_COLOR,
            )

    def _draw_sector_info_card(
        self,
        *,
        x: int,
        y: int,
        title: str,
        lines: list[str],
        accent: tuple[int, int, int],
    ) -> None:
        card_h = 26 + len(lines) * SECTOR_BOARD_CARD_LINE_STEP
        self._draw_rect(
            x,
            y,
            SECTOR_BOARD_CARD_W,
            card_h,
            fill=SECTOR_BOARD_CARD_FILL,
            border=accent,
            border_width=2,
        )
        self._draw_text(
            title,
            x + SECTOR_BOARD_CARD_LINE_X,
            y + SECTOR_BOARD_CARD_TITLE_Y,
            small=True,
            color=SECTOR_BOARD_HUD_PRIMARY_COLOR,
        )
        for idx, line in enumerate(lines):
            self._draw_text(
                _truncate_ui_text(line, max_len=18),
                x + SECTOR_BOARD_CARD_LINE_X,
                y + SECTOR_BOARD_CARD_LINE_Y + idx * SECTOR_BOARD_CARD_LINE_STEP,
                small=True,
                color=SECTOR_BOARD_HUD_SECONDARY_COLOR,
            )

    def _draw_sector_pending_badge(
        self,
        *,
        x: int,
        y: int,
        pending_count: int,
    ) -> None:
        badge_text = _sector_board_pending_badge_text(pending_count)
        if not badge_text:
            return
        self._draw_rect(
            x,
            y,
            SECTOR_BOARD_PENDING_BADGE_SIZE,
            SECTOR_BOARD_PENDING_BADGE_SIZE,
            fill=(171, 108, 187),
            border=(234, 221, 242),
            border_width=1,
        )
        self._draw_text(
            badge_text,
            x + 3,
            y - 1,
            small=True,
            color=(18, 24, 34),
        )

    def _draw_loadout_chips(
        self,
        *,
        x: int,
        y: int,
        pending: Mapping[str, int],
    ) -> None:
        self._loadout_chip_hitboxes = []
        for idx, (token, ap_name) in enumerate(_pending_loadout_entries(pending)):
            chip_x = x + idx * (
                COMMAND_CONSOLE_LOADOUT_CHIP_W + COMMAND_CONSOLE_LOADOUT_CHIP_GAP
            )
            accent = (126, 196, 134) if token != "empty" and not token.startswith("+") else (115, 136, 168)
            fill = (34, 52, 66) if token != "empty" else (40, 48, 64)
            self._draw_rect(
                chip_x,
                y,
                COMMAND_CONSOLE_LOADOUT_CHIP_W,
                COMMAND_CONSOLE_LOADOUT_CHIP_H,
                fill=fill,
                border=accent,
                border_width=1,
            )
            self._draw_text(
                _truncate_ui_text(token, max_len=12),
                chip_x + 8,
                y + 4,
                small=True,
                color=(220, 230, 245),
            )
            if ap_name is not None:
                self._loadout_chip_hitboxes.append(
                    (
                        chip_x,
                        y,
                        COMMAND_CONSOLE_LOADOUT_CHIP_W,
                        COMMAND_CONSOLE_LOADOUT_CHIP_H,
                        ap_name,
                    )
                )

    def _draw_console_sector_chips(
        self,
        *,
        x: int,
        y: int,
        live_sector_name: str | None,
        target_sector_name: str | None,
        pinned_sector_name: str | None,
    ) -> None:
        chip_styles = {
            "LIVE": ((36, 64, 52), (126, 196, 134)),
            "TARGET": ((74, 64, 38), (212, 188, 122)),
            "PIN": ((62, 46, 82), (205, 154, 228)),
        }
        for idx, token in enumerate(
            _command_console_sector_tokens(
                live_sector_name=live_sector_name,
                target_sector_name=target_sector_name,
                pinned_sector_name=pinned_sector_name,
            )
        ):
            chip_x = x + idx * (
                COMMAND_CONSOLE_SECTOR_CHIP_W + COMMAND_CONSOLE_SECTOR_CHIP_GAP
            )
            label = token.split(" ", 1)[0]
            fill, border = chip_styles.get(label, ((40, 48, 64), (115, 136, 168)))
            self._draw_rect(
                chip_x,
                y,
                COMMAND_CONSOLE_SECTOR_CHIP_W,
                COMMAND_CONSOLE_SECTOR_CHIP_H,
                fill=fill,
                border=border,
                border_width=1,
            )
            self._draw_text(
                _truncate_ui_text(token, max_len=15),
                chip_x + 8,
                y + 3,
                small=True,
                color=(232, 238, 249),
            )

    def _draw_command_stage_badge(
        self,
        *,
        x: int,
        y: int,
        live_sector_name: str | None,
        target_sector_name: str | None,
        pinned_sector_name: str | None,
    ) -> None:
        status, _detail = _command_console_stage_status(
            live_sector_name=live_sector_name,
            target_sector_name=target_sector_name,
            pinned_sector_name=pinned_sector_name,
        )
        fills = {
            "ON TARGET": ((92, 78, 42), (212, 188, 122)),
            "ARMED": ((36, 64, 52), (126, 196, 134)),
            "TRACKING": ((62, 46, 82), (205, 154, 228)),
            "STAGING": ((38, 56, 78), (112, 176, 204)),
        }
        fill, border = fills.get(status, ((40, 48, 64), (115, 136, 168)))
        self._draw_rect(
            x,
            y,
            COMMAND_CONSOLE_STAGE_BADGE_W,
            COMMAND_CONSOLE_STAGE_BADGE_H,
            fill=fill,
            border=border,
            border_width=1,
        )
        self._draw_text(
            status,
            x + 10,
            y + 4,
            small=True,
            color=(232, 238, 249),
        )

    def _draw_sector_board_hud(
        self,
        *,
        x: int,
        y: int,
        hovered_cell: _SectorBoardCell | None,
        pinned: bool,
        pressure_levels: Mapping[int, int],
        max_t: int,
        timestep: int,
        t_star: int,
        start_t: int,
        end_t: int,
        window_start: int,
        window_end: int,
        live_cell_name: str | None,
        pending_count: int,
        command_focus_timestep: int | None = None,
        command_focus_timesteps: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        hud_x = x + SECTOR_BOARD_HUD_X_OFFSET
        sections = _sector_board_hud_sections(
            hovered_cell=hovered_cell,
            pressure_levels=pressure_levels,
            max_t=max_t,
            timestep=timestep,
            t_star=t_star,
            start_t=start_t,
            end_t=end_t,
            window_start=window_start,
            window_end=window_end,
            live_cell_name=live_cell_name,
            pending_count=pending_count,
            command_focus_timestep=command_focus_timestep,
            command_focus_timesteps=command_focus_timesteps,
            pinned=pinned,
        )
        card_positions = (
            (hud_x, y + SECTOR_BOARD_CARD_ROW_1_Y),
            (hud_x + SECTOR_BOARD_CARD_W + SECTOR_BOARD_CARD_GAP_X, y + SECTOR_BOARD_CARD_ROW_1_Y),
            (hud_x, y + SECTOR_BOARD_CARD_ROW_2_Y),
            (hud_x + SECTOR_BOARD_CARD_W + SECTOR_BOARD_CARD_GAP_X, y + SECTOR_BOARD_CARD_ROW_2_Y),
        )
        for idx, ((title, lines), (card_x, card_y)) in enumerate(
            zip(sections, card_positions)
        ):
            self._draw_sector_info_card(
                x=card_x,
                y=card_y,
                title=title,
                lines=lines,
                accent=SECTOR_BOARD_HUD_CARD_ACCENTS[idx],
            )

    def _draw_timeline(
        self,
        *,
        timestep: int,
        t_star: int,
        mode: str,
        window_size: int,
        history_atoms: object,
        timeline_span: int,
        pressure_levels: Mapping[int, int] | None = None,
        linked_range: tuple[int, int] | None = None,
    ) -> None:
        history_counts = history_counts_by_t(history_atoms)
        observed_pressure: Mapping[int, int] = pressure_levels or {}
        window_start, window_end = _objective_window_bounds(
            mode=mode,
            t_star=t_star,
            window_size=window_size,
        )
        max_t = max([0, timestep, t_star, *history_counts.keys()])
        start_t, end_t = _timeline_window_bounds(
            timestep=timestep,
            t_star=t_star,
            history_counts=history_counts,
            span=timeline_span,
        )
        cols = max(1, end_t - start_t + 1)
        cell_w = 26
        x0 = 24
        y0 = 130
        self._draw_text("Timeline sectors (t):", x0, y0 - 26)
        for idx, t in enumerate(range(start_t, end_t + 1)):
            x = x0 + idx * (cell_w + 3)
            fill = (34, 44, 62)
            if window_start <= t <= window_end:
                fill = (57, 63, 76)
            if t == t_star and t == timestep:
                fill = (128, 110, 56)
            elif t == t_star:
                fill = (92, 76, 35)
            elif t == timestep:
                fill = (44, 80, 110)
            if history_counts.get(t, 0) > 0:
                # Slightly brighter when interventions happened at t.
                fill = tuple(min(255, c + 30) for c in fill)
            border = (70, 85, 110)
            border_w = 1
            if _range_contains_t(linked_range, t):
                border = (128, 196, 166)
                border_w = 2
            self._draw_rect(
                x,
                y0,
                cell_w,
                30,
                fill=fill,
                border=border,
                border_width=border_w,
            )
            level = observed_pressure.get(t)
            if level is not None:
                self._draw_pressure_band(x=x, y0=y0, cell_w=cell_w, level=level)
            mark = _timeline_mark(t, timestep, t_star)
            if mark:
                self._draw_text(mark, x + 9, y0 - 16, small=True, color=(196, 212, 236))
            self._draw_text(str(t), x + 7, y0 + 8, small=True)
            self._draw_text(
                _pressure_token(observed_pressure.get(t)),
                x + 4,
                y0 + 38,
                small=True,
                color=(176, 191, 216),
            )
            self._draw_text(
                _edits_token(history_counts.get(t)),
                x + 4,
                y0 + 52,
                small=True,
                color=(176, 191, 216),
            )
        self._draw_text("marks: N=now, T=target, B=both", x0, y0 + 66, small=True)
        self._draw_text(
            "P=row pressure (0..10), E=row edits per t, mint border=board hover link",
            x0,
            y0 + 80,
            small=True,
        )
        self._draw_text(
            f"objective window: t={window_start}..{window_end}",
            x0,
            y0 + 96,
            small=True,
            color=(176, 191, 216),
        )
        self._draw_text(
            f"window t={start_t}..{end_t} (span={cols}, [ / ] zoom)",
            x0,
            y0 + 112,
            small=True,
            color=(176, 191, 216),
        )
        self._draw_text(
            _truncate_ui_text(
                _objective_window_pressure_summary(
                    pressure_levels=observed_pressure,
                    window_start=window_start,
                    window_end=window_end,
                ),
                max_len=62,
            ),
            x0,
            y0 + 126,
            small=True,
            color=(176, 191, 216),
        )
        minimap = _build_timeline_minimap(
            max_t=max_t,
            start_t=start_t,
            end_t=end_t,
            timestep=timestep,
            t_star=t_star,
            window_start=window_start,
            window_end=window_end,
            history_counts=history_counts,
            pressure_levels=observed_pressure,
        )
        self._draw_text(
            f"minimap: {minimap}",
            x0,
            y0 + 140,
            small=True,
            color=(176, 191, 216),
        )
        if t_star < start_t:
            self._draw_text(
                "target t* is left of view (press ] to widen or advance time)",
                x0 + 360,
                y0 + 112,
                small=True,
                color=(214, 194, 138),
            )
        elif t_star > end_t:
            self._draw_text(
                "target t* is right of view (press ] to widen or advance time)",
                x0 + 360,
                y0 + 112,
                small=True,
                color=(214, 194, 138),
            )

    def _draw_sector_board(
        self,
        *,
        max_t: int,
        timestep: int,
        t_star: int,
        start_t: int,
        end_t: int,
        window_start: int,
        window_end: int,
        history_counts: Mapping[int, int],
        pressure_levels: Mapping[int, int],
        command_focus_timestep: int | None = None,
        command_focus_timesteps: tuple[int, ...] | list[int] | None = None,
        pending_count: int = 0,
        mouse_pos: tuple[int, int] | None = None,
    ) -> None:
        x = 620
        y = 390
        w = 540
        h = 178
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(24, 34, 50),
            border=(102, 124, 156),
            border_width=2,
        )
        self._draw_text("Sector board (sampled full horizon):", x + 14, y + 10, small=True)
        self._draw_sector_board_legend(x=x, y=y)
        cells = _build_sector_board_cells(
            max_t=max_t,
            timestep=timestep,
            t_star=t_star,
            start_t=start_t,
            end_t=end_t,
            window_start=window_start,
            window_end=window_end,
            history_counts=history_counts,
            pressure_levels=pressure_levels,
            focus_timestep=command_focus_timestep,
            focus_timesteps=command_focus_timesteps,
        )
        cell_size = 18
        gap = 4
        board_x = x + 36
        board_y = y + 52
        hitboxes: list[tuple[int, int, int, int, _SectorBoardCell]] = []
        for col_idx in range(SECTOR_BOARD_COLS):
            col_label = _sector_board_col_label(col_idx)
            self._draw_text(
                col_label,
                board_x + col_idx * (cell_size + gap) + 5,
                board_y - 16,
                small=True,
                color=(176, 191, 216),
            )
        hovered_cell: _SectorBoardCell | None = None
        live_cell: _SectorBoardCell | None = None
        target_cell: _SectorBoardCell | None = None
        pinned_cell: _SectorBoardCell | None = None
        for cell in cells:
            cx = board_x + cell.col * (cell_size + gap)
            cy = board_y + cell.row * (cell_size + gap)
            hitboxes.append((cx, cy, cell_size, cell_size, cell))
            is_live_cell = _range_contains_t((cell.start_t, cell.end_t), timestep)
            if is_live_cell:
                live_cell = cell
            if _range_contains_t((cell.start_t, cell.end_t), t_star):
                target_cell = cell
            if self._pinned_sector_coords == (cell.row, cell.col):
                pinned_cell = cell
            if cell.col == 0:
                self._draw_text(
                    str(cell.row + 1),
                    board_x - 18,
                    cy + 2,
                    small=True,
                    color=(176, 191, 216),
                )
            fill = (38, 48, 66)
            if cell.in_objective_window:
                fill = (58, 65, 80)
            if cell.pressure_level is not None:
                fill = _sector_pressure_fill(cell.pressure_level)
                if cell.in_objective_window:
                    fill = tuple(min(255, channel + 18) for channel in fill)
            if cell.edits > 0:
                fill = tuple(min(255, channel + 10) for channel in fill)
            border = (76, 92, 118)
            if cell.in_viewport:
                border = (164, 184, 214)
            if cell.focus_age == 0:
                border = (126, 196, 134)
            elif cell.focus_age == 1:
                border = (112, 176, 204)
            elif cell.focus_age == 2:
                border = (98, 148, 188)
            if cell.marker:
                border = (212, 188, 122)
            rect = self.pg.Rect(cx, cy, cell_size, cell_size)
            if mouse_pos is not None and rect.collidepoint(mouse_pos):
                hovered_cell = cell
                border = (230, 220, 162)
            if self._pinned_sector_coords == (cell.row, cell.col):
                border = (205, 154, 228)
            self._draw_rect(
                cx,
                cy,
                cell_size,
                cell_size,
                fill=fill,
                border=border,
                border_width=2 if cell.marker else 1,
            )
            glyph = _sector_board_cell_glyph(cell)
            if glyph != ".":
                self._draw_text(
                    glyph,
                    cx + 5,
                    cy + 2,
                    small=True,
                    color=(18, 24, 34),
                )
            if is_live_cell and pending_count > 0:
                self._draw_sector_pending_badge(
                    x=cx + cell_size - SECTOR_BOARD_PENDING_BADGE_OFFSET_X,
                    y=cy - SECTOR_BOARD_PENDING_BADGE_OFFSET_Y,
                    pending_count=pending_count,
                )

        self._sector_board_hitboxes = hitboxes
        active_cell = pinned_cell if pinned_cell is not None else hovered_cell
        self._hovered_sector_range = (
            None
            if active_cell is None
            else (int(active_cell.start_t), int(active_cell.end_t))
        )
        live_cell_name = (
            None
            if live_cell is None
            else _sector_board_cell_name(row=live_cell.row, col=live_cell.col)
        )
        self._live_sector_name = live_cell_name
        self._target_sector_name = (
            None
            if target_cell is None
            else _sector_board_cell_name(row=target_cell.row, col=target_cell.col)
        )
        self._pinned_sector_name = (
            None
            if pinned_cell is None
            else _sector_board_cell_name(row=pinned_cell.row, col=pinned_cell.col)
        )
        self._pinned_sector_range = (
            None
            if pinned_cell is None
            else (int(pinned_cell.start_t), int(pinned_cell.end_t))
        )
        self._draw_sector_board_hud(
            x=x,
            y=y,
            hovered_cell=active_cell,
            pinned=pinned_cell is not None,
            pressure_levels=pressure_levels,
            max_t=max_t,
            timestep=timestep,
            t_star=t_star,
            start_t=start_t,
            end_t=end_t,
            window_start=window_start,
            window_end=window_end,
            live_cell_name=live_cell_name,
            pending_count=pending_count,
            command_focus_timestep=command_focus_timestep,
            command_focus_timesteps=command_focus_timesteps,
        )

    def _draw_help_overlay(self) -> None:
        lines = _help_overlay_lines()
        x = 620
        y = 350
        w = 540
        h = 28 + len(lines) * 28
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(25, 34, 48),
            border=(115, 136, 168),
            border_width=2,
        )
        for idx, line in enumerate(lines):
            color = (229, 236, 250) if idx == 0 else (206, 216, 234)
            self._draw_text(
                line,
                x + 16,
                y + 30 + idx * 28,
                color=color,
                small=(idx != 0),
                title=False,
            )

    def _draw_observation_inspector(
        self,
        *,
        last_obs: dict[str, object] | None,
        timestep: int,
        instance: "GF01Instance",
        objective_text: str,
    ) -> None:
        lines = _observation_inspector_lines(
            _canonical_exposure_payload(
                last_obs=last_obs,
                timestep=timestep,
                instance=instance,
                objective_text=objective_text,
            )
        )
        x = 620
        y = 580
        w = 540
        h = 30 + len(lines) * 22
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(24, 34, 50),
            border=(115, 136, 168),
            border_width=2,
        )
        for idx, line in enumerate(lines):
            color = (229, 236, 250) if idx == 0 else (206, 216, 234)
            self._draw_text(
                line,
                x + 14,
                y + 10 + idx * 22,
                small=True,
                color=color,
            )

    def _draw_onboarding_strip(self, *, timestep: int) -> None:
        lines = _onboarding_strip_lines(timestep)
        if not lines:
            return
        x = 620
        y = 96
        w = 540
        h = 80
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(27, 38, 56),
            border=(115, 136, 168),
            border_width=2,
        )
        self._draw_text(lines[0], x + 14, y + 12, small=True, color=(229, 236, 250))
        self._draw_text(lines[1], x + 14, y + 40, small=True, color=(206, 216, 234))

    def _draw_wave_pressure_strip(
        self,
        *,
        timestep: int,
        label: str,
        filled: int,
        fill_color: tuple[int, int, int],
        trend: str,
        pressure_levels: Mapping[int, int] | None = None,
    ) -> None:
        x = 620
        y = 184
        w = 540
        h = 86
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(24, 34, 50),
            border=(102, 124, 156),
            border_width=2,
        )
        self._draw_text("Sector Wave Strip (observed):", x + 14, y + 10, small=True)
        cell_x = x + 270
        cell_y = y + 10
        for i in range(10):
            active = i < max(0, min(10, int(filled)))
            self._draw_rect(
                cell_x + i * 22,
                cell_y,
                18,
                18,
                fill=fill_color if active else (37, 47, 66),
                border=(92, 108, 132),
            )
        self._draw_text(label, x + 14, y + 36, small=True, color=(198, 212, 234))

        self._wave_strip.update_history(timestep=timestep, trend=trend)
        hot_summary = _format_top_pressure_summary(pressure_levels or {}, max_items=3)
        self._draw_text(
            _truncate_ui_text(
                "Recent wave trends: "
                + self._wave_strip.trail_text()
                + " | Hot sectors: "
                + hot_summary,
                max_len=90,
            ),
            x + 14,
            y + 58,
            small=True,
            color=(176, 191, 216),
        )

    def _draw_command_response_lane(self, *, x: int, y: int) -> None:
        w = 540
        h = 96
        self._draw_rect(
            x,
            y,
            w,
            h,
            fill=(24, 34, 50),
            border=(102, 124, 156),
            border_width=2,
        )
        self._draw_text("Command -> Sector response (observed):", x + 14, y + 10, small=True)
        for idx, line in enumerate(self._command_response_trail.lines()):
            self._draw_text(
                line,
                x + 14,
                y + 34 + idx * 18,
                small=True,
                color=(198, 212, 234),
            )

    def _draw_controls(
        self,
        *,
        timestep: int,
        input_aps: list[str],
        all_input_aps: list[str],
        pending: dict[str, int],
        y_start: int,
        page: int,
        page_size: int,
        group_filter: str | None,
        collapse_rows: bool,
    ) -> tuple[list[_Button], int, int, int]:
        buttons: list[_Button] = []
        visible_aps, page, total_pages = _paginate_input_aps(
            input_aps,
            page=page,
            page_size=page_size,
        )
        panel_h = max(
            COMMAND_CONSOLE_MIN_H,
            COMMAND_CONSOLE_ROWS_Y + 32 + len(visible_aps) * 40,
        )
        panel_y = y_start - COMMAND_CONSOLE_Y_OFFSET
        accent = (205, 154, 228) if self._pinned_sector_name else (102, 124, 156)
        self._draw_rect(
            COMMAND_CONSOLE_X,
            panel_y,
            COMMAND_CONSOLE_W,
            panel_h,
            fill=(24, 34, 50),
            border=accent,
            border_width=2,
        )
        self._draw_text("Command console", 24, panel_y + COMMAND_CONSOLE_TITLE_Y)
        self._draw_command_stage_badge(
            x=COMMAND_CONSOLE_X + COMMAND_CONSOLE_STAGE_BADGE_X,
            y=panel_y + COMMAND_CONSOLE_STAGE_BADGE_Y,
            live_sector_name=self._live_sector_name,
            target_sector_name=self._target_sector_name,
            pinned_sector_name=self._pinned_sector_name,
        )
        self._draw_console_sector_chips(
            x=COMMAND_CONSOLE_X + COMMAND_CONSOLE_SECTOR_CHIP_X,
            y=panel_y + COMMAND_CONSOLE_SECTOR_CHIP_Y,
            live_sector_name=self._live_sector_name,
            target_sector_name=self._target_sector_name,
            pinned_sector_name=self._pinned_sector_name,
        )
        console_lines = _command_console_lines(
            timestep=timestep,
            live_sector_name=self._live_sector_name,
            target_sector_name=self._target_sector_name,
            pinned_sector_name=self._pinned_sector_name,
            pinned_sector_range=self._pinned_sector_range,
        )
        self._draw_text(
            console_lines[0],
            24,
            panel_y + COMMAND_CONSOLE_INFO_1_Y,
            small=True,
            color=(198, 212, 234),
        )
        self._draw_text(
            console_lines[1],
            24,
            panel_y + COMMAND_CONSOLE_INFO_2_Y,
            small=True,
            color=(176, 191, 216),
        )
        self._draw_text(
            "Queued loadout",
            24,
            panel_y + COMMAND_CONSOLE_LOADOUT_LABEL_Y,
            small=True,
            color=(198, 212, 234),
        )
        self._draw_loadout_chips(
            x=24,
            y=panel_y + COMMAND_CONSOLE_LOADOUT_Y,
            pending=pending,
        )
        group_label = "ALL" if group_filter is None else group_filter
        collapse_label = "ON" if collapse_rows else "OFF"
        self._draw_text(
            f"AP page {page + 1}/{total_pages} | page size={page_size}  "
            f"group={group_label} collapse={collapse_label} "
            "(Left/Right, +/-, G group, C collapse)",
            24,
            panel_y + COMMAND_CONSOLE_PAGE_Y,
            small=True,
        )
        self._draw_text(
            _summarize_visible_ap_groups(visible_aps),
            24,
            panel_y + COMMAND_CONSOLE_GROUPS_Y,
            small=True,
            color=(176, 191, 216),
        )
        if collapse_rows:
            self._draw_text(
                "Collapsed map rows:",
                420,
                panel_y + COMMAND_CONSOLE_PAGE_Y,
                small=True,
                color=(176, 191, 216),
            )
            for idx, line in enumerate(_group_rows_for_controls(all_input_aps, pending)):
                self._draw_text(
                    line,
                    420,
                    panel_y + COMMAND_CONSOLE_GROUPS_Y + idx * 18,
                    small=True,
                    color=(176, 191, 216),
                )
        if not visible_aps:
            self._draw_text(
                "No AP rows visible (press G to expand one group).",
                24,
                panel_y + COMMAND_CONSOLE_ROWS_Y,
                small=True,
                color=(176, 191, 216),
            )
            return buttons, page, total_pages, 0
        for idx, ap in enumerate(visible_aps):
            y = panel_y + COMMAND_CONSOLE_ROWS_Y + idx * 40
            row_status, _row_detail = _command_row_status(
                ap=ap,
                pending=pending,
                live_sector_name=self._live_sector_name,
                target_sector_name=self._target_sector_name,
                pinned_sector_name=self._pinned_sector_name,
            )
            row_palette = {
                "TARGET": ((80, 66, 34), (212, 188, 122)),
                "ARMED": ((34, 60, 48), (126, 196, 134)),
                "QUEUED": ((42, 58, 86), (130, 168, 220)),
                "READY": ((30, 42, 58), (96, 118, 150)),
                "IDLE": ((24, 30, 42), (72, 88, 110)),
            }
            card_fill, card_border = row_palette.get(
                row_status, ((30, 42, 58), (96, 118, 150))
            )
            self._draw_rect(
                16,
                y - 2,
                COMMAND_CONSOLE_ROW_CARD_W,
                COMMAND_CONSOLE_ROW_CARD_H,
                fill=card_fill,
                border=card_border,
                border_width=1,
            )
            self._draw_text(ap, 28, y + 6)
            self._draw_rect(
                COMMAND_CONSOLE_ROW_BADGE_X,
                y + 3,
                COMMAND_CONSOLE_ROW_BADGE_W,
                COMMAND_CONSOLE_ROW_BADGE_H,
                fill=(22, 28, 40),
                border=card_border,
                border_width=1,
            )
            self._draw_text(
                row_status,
                COMMAND_CONSOLE_ROW_BADGE_X + 10,
                y + 5,
                small=True,
                color=(220, 230, 245),
            )

            for choice_idx, bit in enumerate((0, 1)):
                x = 220 + choice_idx * 70
                selected = pending.get(ap) == bit
                fill = (42, 100, 62) if selected else (40, 48, 64)
                self._draw_rect(x, y, 56, 28, fill=fill)
                self._draw_text(str(bit), x + 22, y + 5)
                buttons.append(_Button(ap=ap, value=bit, x=x, y=y, w=56, h=28))

            clear_selected = ap in pending
            fill = (120, 70, 60) if clear_selected else (52, 52, 60)
            self._draw_rect(300, y, 82, 28, fill=fill)
            self._draw_text("clear", 316, y + 6, small=True)
            buttons.append(_Button(ap=ap, value=-1, x=300, y=y, w=82, h=28))
        return buttons, page, total_pages, len(visible_aps)

    def choose_action(
        self,
        *,
        last_obs: dict[str, object] | None,
        timestep: int,
        instance: "GF01Instance",
        objective_text: str,
    ) -> dict[str, int] | None:
        pending: dict[str, int] = {}
        current_buttons: list[_Button] = []
        page = 0
        page_size = 10
        timeline_span = 24
        input_aps_all = list(instance.automaton.input_aps)
        group_keys = list(_grouped_input_aps(input_aps_all).keys())
        group_filter: str | None = None
        collapse_rows = False
        if timestep == 0:
            self._previous_observed_y_t = None
            self._last_committed_action_summary = None
            self._last_committed_t = None
            self._wave_strip.reset()
            self._sector_pressure_history.reset()
            self._command_response_trail.reset()
            self._command_focus_timesteps = []
            self._hovered_sector_range = None
            self._sector_board_hitboxes = []
            self._loadout_chip_hitboxes = []
            self._pinned_sector_coords = None
            self._live_sector_name = None
            self._target_sector_name = None
            self._pinned_sector_name = None
            self._pinned_sector_range = None
            self._show_help_overlay = True
            self._show_observation_inspector = False
        previous_y_t = self._previous_observed_y_t
        current_y_t = _normalize_binary_map(
            None if last_obs is None else last_obs.get("y_t", {})
        )
        if last_obs is not None:
            observed_t = max(0, int(timestep) - 1)
            if self._last_committed_t is not None and self._last_committed_t <= int(
                timestep
            ):
                observed_t = max(0, int(self._last_committed_t))
            self._sector_pressure_history.record(
                timestep=observed_t,
                y_t=current_y_t,
            )
        delta_summary = _describe_output_delta(previous_y_t, current_y_t)
        wave_label, wave_filled, wave_fill, wave_trend = _wave_pressure_strip_state(
            previous_y_t, current_y_t
        )
        if current_y_t:
            self._previous_observed_y_t = dict(current_y_t)
        if (
            last_obs is not None
            and self._last_committed_action_summary is not None
            and self._last_committed_t == timestep - 1
        ):
            self._command_response_trail.record(
                timestep=int(self._last_committed_t),
                command=self._last_committed_action_summary,
                response_delta=delta_summary,
            )
        pressure_levels = self._sector_pressure_history.levels()
        while True:
            visible_pool = _control_visible_pool(
                input_aps_all,
                group_filter=group_filter,
                collapse_rows=collapse_rows,
            )
            visible_aps, page, _ = _paginate_input_aps(
                visible_pool,
                page=page,
                page_size=page_size,
            )
            self.clock.tick(30)
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:  # pragma: no cover - UI event
                    return None
                if event.type == self.pg.KEYDOWN:
                    if event.key in (self.pg.K_RETURN, self.pg.K_KP_ENTER):
                        self._last_committed_action_summary = (
                            _summarize_committed_action(pending)
                        )
                        self._last_committed_t = int(timestep)
                        self._record_command_focus(int(timestep))
                        return dict(pending)
                    if event.key == self.pg.K_ESCAPE:
                        self._last_committed_action_summary = (
                            _summarize_committed_action({})
                        )
                        self._last_committed_t = int(timestep)
                        self._record_command_focus(int(timestep))
                        return {}
                    if event.key == self.pg.K_BACKSPACE:
                        pending.clear()
                    if event.key == self.pg.K_h:
                        self._show_help_overlay = not self._show_help_overlay
                    if event.key == self.pg.K_i:
                        self._show_observation_inspector = (
                            not self._show_observation_inspector
                        )
                    if event.key in (self.pg.K_LEFT, self.pg.K_PAGEUP):
                        page = max(0, page - 1)
                    if event.key in (self.pg.K_RIGHT, self.pg.K_PAGEDOWN):
                        # Clamp later using pagination helper.
                        page += 1
                    if event.key in (self.pg.K_EQUALS, self.pg.K_PLUS, self.pg.K_KP_PLUS):
                        page_size = _clamp_page_size(page_size + 1)
                    if event.key in (self.pg.K_MINUS, self.pg.K_UNDERSCORE, self.pg.K_KP_MINUS):
                        page_size = _clamp_page_size(page_size - 1)
                    if event.key == self.pg.K_LEFTBRACKET:
                        timeline_span = _clamp_timeline_span(timeline_span - 4)
                    if event.key == self.pg.K_RIGHTBRACKET:
                        timeline_span = _clamp_timeline_span(timeline_span + 4)
                    if event.key == self.pg.K_g and group_keys:
                        if group_filter is None:
                            group_filter = group_keys[0]
                        else:
                            try:
                                idx = group_keys.index(group_filter) + 1
                            except ValueError:
                                idx = 0
                            group_filter = group_keys[idx] if idx < len(group_keys) else None
                        page = 0
                    if event.key == self.pg.K_c:
                        collapse_rows = not collapse_rows
                        page = 0
                    key_to_index = {
                        self.pg.K_1: 0,
                        self.pg.K_2: 1,
                        self.pg.K_3: 2,
                        self.pg.K_4: 3,
                        self.pg.K_5: 4,
                        self.pg.K_6: 5,
                        self.pg.K_7: 6,
                        self.pg.K_8: 7,
                        self.pg.K_9: 8,
                        self.pg.K_0: 9,
                    }
                    index = key_to_index.get(event.key, None)
                    if index is not None and index < len(visible_aps):
                        _cycle_pending_bit(pending, visible_aps[index])
                if event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    consumed = False
                    for button in current_buttons:
                        if not button.contains(mx, my):
                            continue
                        if button.value == -1:
                            pending.pop(button.ap, None)
                        else:
                            pending[button.ap] = button.value
                        consumed = True
                        break
                    if consumed:
                        continue
                    for x0, y0, w0, h0, ap_name in self._loadout_chip_hitboxes:
                        if not (x0 <= mx <= x0 + w0 and y0 <= my <= y0 + h0):
                            continue
                        pending.pop(ap_name, None)
                        consumed = True
                        break
                    if consumed:
                        continue
                    for x0, y0, w0, h0, cell in self._sector_board_hitboxes:
                        if not (x0 <= mx <= x0 + w0 and y0 <= my <= y0 + h0):
                            continue
                        coords = (cell.row, cell.col)
                        if self._pinned_sector_coords == coords:
                            self._pinned_sector_coords = None
                        else:
                            self._pinned_sector_coords = coords
                        break

            self.screen.fill((18, 24, 34))
            self._draw_text("GF-01-R1 Map-First Visual", 24, 18, title=True)
            self._draw_text(f"t={timestep}  t*={instance.t_star}  mode={instance.mode}", 24, 54)
            self._draw_text(objective_text, 24, 82)
            self._draw_onboarding_strip(timestep=timestep)
            self._draw_wave_pressure_strip(
                timestep=timestep,
                label=wave_label,
                filled=wave_filled,
                fill_color=wave_fill,
                trend=wave_trend,
                pressure_levels=pressure_levels,
            )
            self._draw_command_response_lane(x=620, y=278)

            history_atoms = [] if last_obs is None else last_obs.get("history_atoms", [])
            history_counts = history_counts_by_t(history_atoms)
            max_t = max([0, int(timestep), int(instance.t_star), *history_counts.keys()])
            window_start, window_end = _objective_window_bounds(
                mode=str(instance.mode),
                t_star=int(instance.t_star),
                window_size=int(instance.window_size),
            )
            start_t, end_t = _timeline_window_bounds(
                timestep=int(timestep),
                t_star=int(instance.t_star),
                history_counts=history_counts,
                span=timeline_span,
            )
            self._draw_sector_board(
                max_t=max_t,
                timestep=int(timestep),
                t_star=int(instance.t_star),
                start_t=start_t,
                end_t=end_t,
                window_start=window_start,
                window_end=window_end,
                history_counts=history_counts,
                pressure_levels=pressure_levels,
                command_focus_timestep=self._last_committed_t,
                command_focus_timesteps=self._command_focus_timesteps,
                pending_count=len(pending),
                mouse_pos=self.pg.mouse.get_pos(),
            )
            self._draw_timeline(
                timestep=timestep,
                t_star=int(instance.t_star),
                mode=str(instance.mode),
                window_size=int(instance.window_size),
                history_atoms=history_atoms,
                timeline_span=timeline_span,
                pressure_levels=pressure_levels,
                linked_range=self._hovered_sector_range,
            )
            current_buttons, page, _, _ = self._draw_controls(
                timestep=int(timestep),
                input_aps=visible_pool,
                all_input_aps=input_aps_all,
                pending=pending,
                y_start=286,
                page=page,
                page_size=page_size,
                group_filter=group_filter,
                collapse_rows=collapse_rows,
            )
            status_x = 460
            status_y = 250
            if last_obs is None:
                self._draw_text(
                    "No prior observation yet (episode start).", status_x, status_y
                )
                self._draw_text(
                    _summarize_pending_interventions(pending),
                    status_x,
                    status_y + 28,
                    small=True,
                    color=(192, 209, 232),
                )
            else:
                y_t = last_obs.get("y_t", {})
                effect = str(last_obs.get("effect_status_t", "unknown"))
                bt = int(last_obs.get("budget_t_remaining", instance.budget_timestep))
                ba = int(last_obs.get("budget_a_remaining", instance.budget_atoms))
                effect_text, effect_fill = _effect_status_badge(effect)
                self._draw_text("Observation Summary:", status_x, status_y)
                self._draw_text(
                    _summarize_observed_outputs(y_t), status_x, status_y + 28
                )
                self._draw_text(
                    f"Budget remaining: timesteps={bt}, atoms={ba}",
                    status_x,
                    status_y + 56,
                )
                self._draw_rect(
                    status_x + 430,
                    status_y + 52,
                    240,
                    28,
                    fill=effect_fill,
                    border=(114, 132, 158),
                )
                self._draw_text(
                    effect_text,
                    status_x + 438,
                    status_y + 59,
                    small=True,
                    color=(232, 238, 249),
                )
                delta_y = status_y + 84
                if (
                    self._last_committed_action_summary is not None
                    and self._last_committed_t == timestep - 1
                ):
                    self._draw_text(
                        "Previous command: "
                        f"{self._last_committed_action_summary}",
                        status_x,
                        status_y + 84,
                    )
                    delta_y = status_y + 112
                self._draw_text(delta_summary, status_x, delta_y)
                self._draw_text(
                    _summarize_pending_interventions(pending),
                    status_x,
                    delta_y + 28,
                    small=True,
                    color=(192, 209, 232),
                )

            footer_y = 712
            self._draw_text("Controls:", 24, footer_y, small=True)
            self._draw_text(
                "Click 0/1 to set AP value, clear to unset | 1..9,0 cycle APs | "
                "Enter=commit | Esc=skip | Backspace=clear all | "
                "Left/Right=AP page | +/-=AP density | [ ]=timeline zoom | "
                "G=group | C=collapse | H=help | I=inspector",
                100,
                footer_y,
                small=True,
            )
            if self._show_help_overlay:
                self._draw_help_overlay()
            if self._show_observation_inspector:
                self._draw_observation_inspector(
                    last_obs=last_obs,
                    timestep=timestep,
                    instance=instance,
                    objective_text=objective_text,
                )
            self.pg.display.flip()


def _session() -> _R1PygameSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = _R1PygameSession()
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
