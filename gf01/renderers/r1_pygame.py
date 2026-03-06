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


_SESSION: "_R1PygameSession | None" = None
UI_TEXT_MIN_TRUNCATE_LEN = 4
COMMAND_RESPONSE_LINE_MAX_LEN = 84
COMMAND_RESPONSE_MAX_ENTRIES = 4
COMMAND_RESPONSE_MAX_VISIBLE = 3
SECTOR_PRESSURE_BANDS = 10
SECTOR_PRESSURE_HISTORY_MAX = 256
TIMELINE_MINIMAP_CHARS = 48


@dataclass(frozen=True)
class _Button:
    ap: str
    value: int
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def _paginate_input_aps(
    input_aps: list[str], page: int, page_size: int
) -> tuple[list[str], int, int]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not input_aps:
        return [], 0, 1
    total_pages = (len(input_aps) + page_size - 1) // page_size
    page_clamped = min(max(int(page), 0), total_pages - 1)
    start = page_clamped * page_size
    end = start + page_size
    return input_aps[start:end], page_clamped, total_pages


def _normalize_binary_map(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in payload.items():
        name = str(key)
        try:
            bit = int(value)
        except (TypeError, ValueError):
            continue
        if bit in (0, 1):
            out[name] = bit
    return out


def _describe_output_delta(previous: object, current: object) -> str:
    prev_map = _normalize_binary_map(previous)
    curr_map = _normalize_binary_map(current)
    if not curr_map:
        return "Output delta: unavailable"
    if not prev_map:
        return "Output delta: baseline observation"
    changed: list[str] = []
    for ap in sorted(set(prev_map) | set(curr_map)):
        prev_v = prev_map.get(ap)
        curr_v = curr_map.get(ap)
        if prev_v != curr_v:
            if prev_v is None:
                changed.append(f"{ap}: ? -> {curr_v}")
            elif curr_v is None:
                changed.append(f"{ap}: {prev_v} -> ?")
            else:
                changed.append(f"{ap}: {prev_v} -> {curr_v}")
    if not changed:
        return "Output delta: no observed change"
    return "Output delta: " + ", ".join(changed)


def _cycle_pending_bit(pending: dict[str, int], ap: str) -> None:
    current = pending.get(ap)
    if current is None:
        pending[ap] = 1
        return
    if current == 1:
        pending[ap] = 0
        return
    pending.pop(ap, None)


def _clamp_page_size(value: int, *, minimum: int = 4, maximum: int = 16) -> int:
    return min(max(int(value), int(minimum)), int(maximum))


def _clamp_timeline_span(value: int, *, minimum: int = 8, maximum: int = 48) -> int:
    return min(max(int(value), int(minimum)), int(maximum))


def _truncate_ui_text(
    text: str, *, max_len: int = COMMAND_RESPONSE_LINE_MAX_LEN
) -> str:
    max_len = max(UI_TEXT_MIN_TRUNCATE_LEN, int(max_len))
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _pressure_level_from_observation(y_t: object) -> int | None:
    observed = _normalize_binary_map(y_t)
    if not observed:
        return None
    on_count = sum(1 for bit in observed.values() if bit == 1)
    ratio = on_count / float(len(observed))
    return max(0, min(SECTOR_PRESSURE_BANDS, int(round(ratio * SECTOR_PRESSURE_BANDS))))


def _sector_pressure_fill(level: int) -> tuple[int, int, int]:
    clamped = max(0, min(SECTOR_PRESSURE_BANDS, int(level)))
    ratio = clamped / float(SECTOR_PRESSURE_BANDS)
    return (
        46 + int(90.0 * ratio),
        74 + int(110.0 * ratio),
        112 + int(100.0 * ratio),
    )


def _pressure_token(level: int | None) -> str:
    if level is None:
        return "P."
    clamped = max(0, min(SECTOR_PRESSURE_BANDS, int(level)))
    return f"P{clamped}"


def _edits_token(edits: int | None) -> str:
    if edits is None:
        return "E."
    return f"E{max(0, int(edits))}"


def _top_pressure_sectors(
    pressure_levels: Mapping[int, int], *, max_items: int = 4
) -> list[tuple[int, int]]:
    if not pressure_levels:
        return []
    ranked = sorted(
        (
            (int(t), max(0, min(SECTOR_PRESSURE_BANDS, int(level))))
            for t, level in pressure_levels.items()
        ),
        key=lambda item: (-item[1], -item[0]),
    )
    return ranked[: max(1, int(max_items))]


def _format_top_pressure_summary(
    pressure_levels: Mapping[int, int], *, max_items: int = 4
) -> str:
    top = _top_pressure_sectors(pressure_levels, max_items=max_items)
    if not top:
        return "(none yet)"
    return ", ".join(f"t={t}:{_pressure_token(level)}" for t, level in top)


def _objective_window_pressure_summary(
    *,
    pressure_levels: Mapping[int, int],
    window_start: int,
    window_end: int,
) -> str:
    low = int(window_start)
    high = int(window_end)
    if high < low:
        return "Window pressure: invalid"
    total = high - low + 1
    observed = [
        (
            int(t),
            max(0, min(SECTOR_PRESSURE_BANDS, int(level))),
        )
        for t, level in pressure_levels.items()
        if low <= int(t) <= high
    ]
    if not observed:
        return f"Window pressure: 0/{total} observed"
    peak_t, peak_level = max(observed, key=lambda item: (item[1], item[0]))
    avg = sum(level for _, level in observed) / float(len(observed))
    return (
        f"Window pressure: {len(observed)}/{total} observed, "
        f"peak t={peak_t}:{_pressure_token(peak_level)}, avg={avg:.1f}"
    )


def _build_timeline_minimap(
    *,
    max_t: int,
    start_t: int,
    end_t: int,
    timestep: int,
    t_star: int,
    window_start: int,
    window_end: int,
    history_counts: Mapping[int, int],
    pressure_levels: Mapping[int, int],
    width: int = TIMELINE_MINIMAP_CHARS,
) -> str:
    chars = max(8, int(width))
    horizon = max(0, int(max_t))
    if chars == 1:
        return "N"
    if horizon == 0:
        base = ["." for _ in range(chars)]
        idx = 0
        if int(timestep) == int(t_star):
            base[idx] = "B"
        elif int(timestep) == 0:
            base[idx] = "N"
        elif int(t_star) == 0:
            base[idx] = "T"
        base[idx] = "["
        base[-1] = "]"
        return "".join(base)

    def _sample_t(col: int) -> int:
        return int(round(col * horizon / float(chars - 1)))

    strip: list[str] = []
    for col in range(chars):
        t = _sample_t(col)
        token = "."
        if int(window_start) <= t <= int(window_end):
            token = "w"
        if int(history_counts.get(t, 0)) > 0:
            token = "e"
        if t in pressure_levels:
            token = "p"
        if t == int(timestep) and t == int(t_star):
            token = "B"
        elif t == int(timestep):
            token = "N"
        elif t == int(t_star):
            token = "T"
        strip.append(token)

    start_idx = int(round(max(0, int(start_t)) * (chars - 1) / float(horizon)))
    end_idx = int(round(max(0, int(end_t)) * (chars - 1) / float(horizon)))
    start_idx = max(0, min(chars - 1, start_idx))
    end_idx = max(0, min(chars - 1, end_idx))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    strip[start_idx] = "["
    strip[end_idx] = "]"
    return "".join(strip)


def _timeline_window_bounds(
    *,
    timestep: int,
    t_star: int,
    history_counts: dict[int, int],
    span: int,
) -> tuple[int, int]:
    max_t = max([0, int(timestep), int(t_star), *history_counts.keys()])
    span_clamped = _clamp_timeline_span(span)
    if max_t + 1 <= span_clamped:
        return (0, max_t)

    low = min(int(timestep), int(t_star))
    high = max(int(timestep), int(t_star))
    required = high - low + 1
    if required <= span_clamped:
        pad_total = span_clamped - required
        pad_left = pad_total // 2
        start = max(0, low - pad_left)
        end = start + span_clamped - 1
        if end > max_t:
            end = max_t
            start = max(0, end - span_clamped + 1)
        return (start, end)

    # If now/target cannot both fit, prioritize "now" so current play remains legible.
    start = max(0, min(int(timestep) - span_clamped // 2, max_t - span_clamped + 1))
    end = min(max_t, start + span_clamped - 1)
    return (start, end)


def _objective_window_bounds(
    *, mode: str, t_star: int, window_size: int
) -> tuple[int, int]:
    if str(mode).strip().lower() == "hard":
        t = int(t_star)
        return (t, t)
    start = max(0, int(t_star) - max(0, int(window_size)))
    return (start, int(t_star))


def _help_overlay_lines() -> list[str]:
    return [
        "GF-01-R1 quick help",
        "Goal: commit interventions to reach the visible objective.",
        "Mouse: click 0/1 to set AP, clear to unset AP.",
        "Keys: 1..9,0 cycle AP slots on current page.",
        "Keys: Left/Right page APs | +/- AP density.",
        "Keys: [ / ] timeline zoom (narrow/wide).",
        "Keys: G AP group filter | C collapse/expand map rows.",
        "Keys: I canonical observation inspector toggle.",
        "Keys: Enter commit | Esc skip | Backspace clear all.",
        "Tip: read Previous command, Output delta, and Wave strip together.",
        "Press H to hide/show this panel.",
    ]


def _canonical_exposure_payload(
    *,
    last_obs: dict[str, object] | None,
    timestep: int,
    instance: "GF01Instance",
    objective_text: str,
) -> dict[str, object]:
    mission = {
        "objective_text": str(objective_text),
        "effect_ap": str(instance.effect_ap),
        "t_star": int(instance.t_star),
        "mode": str(instance.mode),
        "budget_timestep": int(instance.budget_timestep),
        "budget_atoms": int(instance.budget_atoms),
        "timestep_prompt": int(timestep),
    }
    if last_obs is None:
        return {"mission": mission, "observation": None}
    canonical_order = [
        "t",
        "y_t",
        "effect_status_t",
        "budget_t_remaining",
        "budget_a_remaining",
        "history_atoms",
        "mode",
        "t_star",
    ]
    observation: dict[str, object] = {}
    for key in canonical_order:
        if key in last_obs:
            observation[key] = last_obs[key]
    return {"mission": mission, "observation": observation}


def _observation_inspector_lines(payload: dict[str, object]) -> list[str]:
    mission = payload.get("mission", {})
    observation = payload.get("observation", None)
    lines = ["Canonical observation inspector (I): I(s)"]
    lines.append(
        _truncate_ui_text(
            "mission: "
            + json.dumps(mission, sort_keys=True, separators=(",", ":")),
            max_len=96,
        )
    )
    if observation is None:
        lines.append("observation: (none yet)")
    else:
        lines.append(
            _truncate_ui_text(
                "observation: "
                + json.dumps(observation, sort_keys=True, separators=(",", ":")),
                max_len=96,
            )
        )
    return lines


def _ap_group_key(ap: str) -> str:
    token = str(ap).strip()
    if not token:
        return "unknown"
    prefix_chars: list[str] = []
    for ch in token:
        if ch.isdigit():
            break
        prefix_chars.append(ch)
    prefix = "".join(prefix_chars).strip("_-")
    if prefix:
        return prefix
    if "_" in token:
        head = token.split("_", 1)[0].strip()
        if head:
            return head
    return token


def _summarize_visible_ap_groups(visible_aps: list[str]) -> str:
    if not visible_aps:
        return "groups: (none)"
    grouped: dict[str, int] = defaultdict(int)
    for ap in visible_aps:
        grouped[_ap_group_key(ap)] += 1
    parts = [f"{k}({grouped[k]})" for k in sorted(grouped)]
    return "groups: " + ", ".join(parts)


def _grouped_input_aps(input_aps: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for ap in input_aps:
        grouped[_ap_group_key(ap)].append(ap)
    return {key: grouped[key] for key in sorted(grouped)}


def _apply_group_filter(input_aps: list[str], group_key: str | None) -> list[str]:
    if group_key is None:
        return list(input_aps)
    selected: list[str] = []
    for ap in input_aps:
        if _ap_group_key(ap) == group_key:
            selected.append(ap)
    return selected


def _control_visible_pool(
    input_aps: list[str],
    *,
    group_filter: str | None,
    collapse_rows: bool,
) -> list[str]:
    if collapse_rows and group_filter is None:
        return []
    return _apply_group_filter(input_aps, group_filter)


def _group_rows_for_controls(
    input_aps: list[str],
    pending: dict[str, int],
    *,
    max_groups: int = 8,
) -> list[str]:
    grouped = _grouped_input_aps(input_aps)
    if not grouped:
        return ["(none)"]
    lines: list[str] = []
    keys = sorted(grouped)
    for key in keys[: max(1, int(max_groups))]:
        aps = grouped[key]
        set_count = sum(1 for ap in aps if ap in pending)
        lines.append(f"{key}: {set_count}/{len(aps)} set")
    hidden = len(keys) - len(lines)
    if hidden > 0:
        lines.append(f"+{hidden} more groups")
    return lines


def _timeline_mark(t: int, timestep: int, t_star: int) -> str:
    marker = timeline_marker_for_t(t, t_now=timestep, t_star=t_star)
    if marker == ".":
        return ""
    return marker


def _summarize_observed_outputs(y_t: object, *, max_names: int = 6) -> str:
    observed = _normalize_binary_map(y_t)
    if not observed:
        return "Observed outputs: unavailable"
    active = sorted(ap for ap, bit in observed.items() if bit == 1)
    total = len(observed)
    if not active:
        return f"Observed outputs: all OFF (0/{total} ON)"
    shown = active[:max(1, int(max_names))]
    joined = ", ".join(shown)
    if len(active) > len(shown):
        joined += f", +{len(active) - len(shown)} more"
    return f"Observed outputs ON: {joined} ({len(active)}/{total})"


def _summarize_pending_interventions(
    pending: dict[str, int], *, max_items: int = 6
) -> str:
    items = [
        (str(ap), int(bit))
        for ap, bit in sorted(pending.items())
        if int(bit) in (0, 1)
    ]
    if not items:
        return "Pending interventions: none selected"
    shown = items[: max(1, int(max_items))]
    body = ", ".join(f"{ap}={bit}" for ap, bit in shown)
    if len(items) > len(shown):
        body += f", +{len(items) - len(shown)} more"
    return "Pending interventions: " + body


def _summarize_committed_action(
    action: dict[str, int], *, max_items: int = 6
) -> str:
    items = [
        (str(ap), int(bit))
        for ap, bit in sorted(action.items())
        if int(bit) in (0, 1)
    ]
    if not items:
        return "no interventions (skip)"
    shown = items[: max(1, int(max_items))]
    body = ", ".join(f"{ap}={bit}" for ap, bit in shown)
    if len(items) > len(shown):
        body += f", +{len(items) - len(shown)} more"
    return body


def _effect_status_badge(effect_status: object) -> tuple[str, tuple[int, int, int]]:
    token = str(effect_status).strip().lower()
    if token == "triggered":
        return ("Objective active", (46, 112, 66))
    if token == "not-triggered":
        return ("Objective not active", (112, 84, 42))
    if token == "unknown":
        return ("Objective status unknown", (64, 72, 90))
    return (f"Objective status: {effect_status}", (64, 72, 90))


class _WaveStripModel:
    def __init__(self) -> None:
        self._trend_history: list[tuple[int, str]] = []

    @staticmethod
    def compute_state(
        previous_y_t: object, current_y_t: object
    ) -> tuple[str, int, tuple[int, int, int], str]:
        current = _normalize_binary_map(current_y_t)
        if not current:
            return ("Wave pressure: awaiting observation", 0, (64, 72, 90), "none")
        prev = _normalize_binary_map(previous_y_t)
        on_count = sum(1 for bit in current.values() if bit == 1)
        total = len(current)
        ratio = on_count / float(total)
        filled = max(0, min(10, int(round(ratio * 10.0))))
        if not prev:
            trend = "baseline"
        else:
            prev_on = sum(1 for bit in prev.values() if bit == 1)
            prev_ratio = prev_on / float(len(prev))
            delta = ratio - prev_ratio
            if delta > 0.05:
                trend = "rising"
            elif delta < -0.05:
                trend = "falling"
            else:
                trend = "steady"
        # Blue-cyan ramp to indicate magnitude without implying good/bad semantics.
        fill = (
            56 + int(80.0 * ratio),
            92 + int(96.0 * ratio),
            136 + int(96.0 * ratio),
        )
        label = f"Wave pressure: {on_count}/{total} active ({trend})"
        return (label, filled, fill, trend)

    def reset(self) -> None:
        self._trend_history = []

    def update_history(self, *, timestep: int, trend: str) -> None:
        if trend == "none":
            return
        entry = (int(timestep), str(trend))
        if not self._trend_history or self._trend_history[-1] != entry:
            self._trend_history.append(entry)
        if len(self._trend_history) > 5:
            self._trend_history = self._trend_history[-5:]

    def trail_text(self) -> str:
        if not self._trend_history:
            return "(none yet)"
        return " | ".join(f"t={t}:{tr}" for t, tr in self._trend_history[-4:])


class _SectorPressureHistoryModel:
    """Store observed sector-pressure levels by timestep for timeline rendering."""

    def __init__(self, *, max_entries: int = SECTOR_PRESSURE_HISTORY_MAX) -> None:
        self._max_entries = max(1, int(max_entries))
        self._levels: OrderedDict[int, int] = OrderedDict()

    def reset(self) -> None:
        self._levels = OrderedDict()

    def record(self, *, timestep: int, y_t: object) -> None:
        level = _pressure_level_from_observation(y_t)
        if level is None:
            return
        t = int(timestep)
        if t in self._levels:
            self._levels.pop(t, None)
        self._levels[t] = level
        while len(self._levels) > self._max_entries:
            self._levels.popitem(last=False)

    def levels(self) -> Mapping[int, int]:
        return MappingProxyType(self._levels)


class _CommandResponseTrailModel:
    """Track recent command->response summaries for rendering.

    `max_entries` controls storage capacity and de-dup window.
    `max_visible_lines` controls how many of the most recent stored entries
    are returned by `lines()` for UI display.
    """

    def __init__(
        self,
        *,
        max_entries: int = COMMAND_RESPONSE_MAX_ENTRIES,
        max_visible_lines: int = COMMAND_RESPONSE_MAX_VISIBLE,
        line_max_len: int = COMMAND_RESPONSE_LINE_MAX_LEN,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_visible_lines = max(1, int(max_visible_lines))
        self._line_max_len = max(UI_TEXT_MIN_TRUNCATE_LEN, int(line_max_len))
        self._entries: list[str] = []

    def reset(self) -> None:
        self._entries = []

    def record(self, *, timestep: int, command: str, response_delta: str) -> None:
        entry = _truncate_ui_text(
            f"t={timestep} | {command} -> {response_delta}",
            max_len=self._line_max_len,
        )
        if self._entries and self._entries[-1] == entry:
            return
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def lines(self) -> list[str]:
        if not self._entries:
            return ["(none yet)"]
        return self._entries[-self._max_visible_lines :]


def _wave_pressure_strip_state(
    previous_y_t: object, current_y_t: object
) -> tuple[str, int, tuple[int, int, int], str]:
    return _WaveStripModel.compute_state(previous_y_t, current_y_t)


def _onboarding_strip_lines(timestep: int) -> list[str]:
    t = int(timestep)
    if t == 0:
        return [
            "Onboarding 1/3: Set one control, then commit.",
            "Use click or 1..9,0. Commit with Enter.",
        ]
    if t == 1:
        return [
            "Onboarding 2/3: Read cause -> effect.",
            "Compare Previous command with Output delta.",
        ]
    if t == 2:
        return [
            "Onboarding 3/3: Adjust based on feedback.",
            "Watch Objective badge, timeline marks, and budget.",
        ]
    return []


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
            self._draw_rect(x, y0, cell_w, 30, fill=fill)
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
            "P=row pressure (0..10), E=row edits per t",
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
        y = 520
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
        self._draw_text("Set interventions for current timestep:", 24, y_start - 26)
        visible_aps, page, total_pages = _paginate_input_aps(
            input_aps,
            page=page,
            page_size=page_size,
        )
        group_label = "ALL" if group_filter is None else group_filter
        collapse_label = "ON" if collapse_rows else "OFF"
        self._draw_text(
            f"AP page {page + 1}/{total_pages} | page size={page_size}  "
            f"group={group_label} collapse={collapse_label} "
            "(Left/Right, +/-, G group, C collapse)",
            24,
            y_start - 4,
            small=True,
        )
        self._draw_text(
            _summarize_visible_ap_groups(visible_aps),
            24,
            y_start + 16,
            small=True,
            color=(176, 191, 216),
        )
        if collapse_rows:
            self._draw_text(
                "Collapsed map rows:",
                420,
                y_start - 4,
                small=True,
                color=(176, 191, 216),
            )
            for idx, line in enumerate(_group_rows_for_controls(all_input_aps, pending)):
                self._draw_text(
                    line,
                    420,
                    y_start + 16 + idx * 18,
                    small=True,
                    color=(176, 191, 216),
                )
        if not visible_aps:
            self._draw_text(
                "No AP rows visible (press G to expand one group).",
                24,
                y_start + 52,
                small=True,
                color=(176, 191, 216),
            )
            return buttons, page, total_pages, 0
        for idx, ap in enumerate(visible_aps):
            y = y_start + 24 + idx * 40
            self._draw_text(ap, 24, y + 8)

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
                        return dict(pending)
                    if event.key == self.pg.K_ESCAPE:
                        self._last_committed_action_summary = (
                            _summarize_committed_action({})
                        )
                        self._last_committed_t = int(timestep)
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
                    for button in current_buttons:
                        if not button.contains(mx, my):
                            continue
                        if button.value == -1:
                            pending.pop(button.ap, None)
                        else:
                            pending[button.ap] = button.value
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
            self._draw_timeline(
                timestep=timestep,
                t_star=int(instance.t_star),
                mode=str(instance.mode),
                window_size=int(instance.window_size),
                history_atoms=history_atoms,
                timeline_span=timeline_span,
                pressure_levels=pressure_levels,
            )
            current_buttons, page, _, _ = self._draw_controls(
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
