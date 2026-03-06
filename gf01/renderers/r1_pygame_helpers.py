"""Deterministic helper types and functions for the GF-01-R1 pygame renderer."""
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
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..semantics import history_counts_by_t, timeline_marker_for_t

if TYPE_CHECKING:
    from ..models import GF01Instance

UI_TEXT_MIN_TRUNCATE_LEN = 4
COMMAND_RESPONSE_LINE_MAX_LEN = 84
COMMAND_RESPONSE_MAX_ENTRIES = 4
COMMAND_RESPONSE_MAX_VISIBLE = 3
SECTOR_PRESSURE_BANDS = 10
SECTOR_PRESSURE_HISTORY_MAX = 256
TIMELINE_MINIMAP_CHARS = 48
SECTOR_BOARD_COLS = 8
SECTOR_BOARD_ROWS = 6


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


@dataclass(frozen=True)
class _SectorBoardCell:
    row: int
    col: int
    start_t: int
    end_t: int
    pressure_level: int | None
    edits: int
    marker: str
    in_viewport: bool
    in_objective_window: bool
    is_command_focus: bool
    focus_age: int | None


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
    t_now = int(timestep)
    t_target = int(t_star)
    view_start = int(start_t)
    view_end = int(end_t)
    win_start = int(window_start)
    win_end = int(window_end)

    if chars == 1:
        return "N"
    if horizon == 0:
        base = ["." for _ in range(chars)]
        marker = _bucket_marker(0, 0, t_now=t_now, t_target=t_target)
        base[0] = "["
        base[-1] = "]"
        if marker:
            marker_idx = 1 if chars > 2 else 0
            base[marker_idx] = marker
        elif win_start <= 0 <= win_end and chars > 2:
            base[1] = "w"
        return "".join(base)

    history_bucket = _bucketize_history_counts(
        history_counts=history_counts,
        max_t=horizon,
        bucket_count=chars,
    )
    pressure_bucket = _bucketize_pressure_levels(
        pressure_levels=pressure_levels,
        max_t=horizon,
        bucket_count=chars,
    )
    strip: list[str] = []
    for idx in range(chars):
        bucket_start, bucket_end = _sector_bucket_bounds(
            index=idx,
            bucket_count=chars,
            max_t=horizon,
        )
        token = "."
        if _ranges_overlap(bucket_start, bucket_end, win_start, win_end):
            token = "w"
        if history_bucket[idx] > 0:
            token = "e"
        if pressure_bucket[idx] is not None:
            token = "p"
        marker = _bucket_marker(bucket_start, bucket_end, t_now=t_now, t_target=t_target)
        if marker:
            token = marker
        strip.append(token)

    start_idx = _bucket_index_for_t(t=view_start, max_t=horizon, bucket_count=chars)
    end_idx = _bucket_index_for_t(t=view_end, max_t=horizon, bucket_count=chars)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    _place_minimap_bracket(strip, index=start_idx, symbol="[", step=1)
    _place_minimap_bracket(strip, index=end_idx, symbol="]", step=-1)
    return "".join(strip)


def _sector_bucket_bounds(
    *, index: int, bucket_count: int, max_t: int
) -> tuple[int, int]:
    horizon = max(1, int(max_t) + 1)
    bucket_n = max(1, int(bucket_count))
    idx = min(max(0, int(index)), bucket_n - 1)
    start = (idx * horizon) // bucket_n
    end = ((idx + 1) * horizon) // bucket_n - 1
    if end < start:
        end = start
    max_step = max(0, int(max_t))
    return (min(start, max_step), min(end, max_step))


def _bucket_index_for_t(*, t: int, max_t: int, bucket_count: int) -> int:
    max_step = max(0, int(max_t))
    bucket_n = max(1, int(bucket_count))
    t_clamped = max(0, min(max_step, int(t)))
    return min(bucket_n - 1, (t_clamped * bucket_n) // (max_step + 1))


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    low_a = min(int(a_start), int(a_end))
    high_a = max(int(a_start), int(a_end))
    low_b = min(int(b_start), int(b_end))
    high_b = max(int(b_start), int(b_end))
    return not (high_a < low_b or low_a > high_b)


def _range_contains_t(bounds: tuple[int, int] | None, t: int) -> bool:
    if bounds is None:
        return False
    low = min(int(bounds[0]), int(bounds[1]))
    high = max(int(bounds[0]), int(bounds[1]))
    return low <= int(t) <= high


def _bucket_marker(bucket_start: int, bucket_end: int, *, t_now: int, t_target: int) -> str:
    now_in = int(bucket_start) <= int(t_now) <= int(bucket_end)
    target_in = int(bucket_start) <= int(t_target) <= int(bucket_end)
    if now_in and target_in:
        return "B"
    if now_in:
        return "N"
    if target_in:
        return "T"
    return ""


def _bucketize_history_counts(
    *,
    history_counts: Mapping[int, int],
    max_t: int,
    bucket_count: int,
) -> list[int]:
    bucket_n = max(1, int(bucket_count))
    edits = [0 for _ in range(bucket_n)]
    for t_key, count in history_counts.items():
        idx = _bucket_index_for_t(t=int(t_key), max_t=max_t, bucket_count=bucket_n)
        edits[idx] += max(0, int(count))
    return edits


def _bucketize_pressure_levels(
    *,
    pressure_levels: Mapping[int, int],
    max_t: int,
    bucket_count: int,
) -> list[int | None]:
    bucket_n = max(1, int(bucket_count))
    pressure: list[int | None] = [None for _ in range(bucket_n)]
    for t_key, level in pressure_levels.items():
        idx = _bucket_index_for_t(t=int(t_key), max_t=max_t, bucket_count=bucket_n)
        clamped = max(0, min(SECTOR_PRESSURE_BANDS, int(level)))
        current = pressure[idx]
        pressure[idx] = clamped if current is None else max(current, clamped)
    return pressure


def _place_minimap_bracket(strip: list[str], *, index: int, symbol: str, step: int) -> None:
    if not strip:
        return
    idx = max(0, min(len(strip) - 1, int(index)))
    marker_tokens = {"N", "T", "B"}
    if strip[idx] not in marker_tokens:
        strip[idx] = symbol
        return
    cursor = idx + (1 if int(step) >= 0 else -1)
    while 0 <= cursor < len(strip):
        if strip[cursor] not in marker_tokens:
            strip[cursor] = symbol
            return
        cursor += 1 if int(step) >= 0 else -1
    strip[idx] = symbol


def _build_sector_board_cells(
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
    focus_timestep: int | None = None,
    focus_timesteps: tuple[int, ...] | list[int] | None = None,
    cols: int = SECTOR_BOARD_COLS,
    rows: int = SECTOR_BOARD_ROWS,
) -> list[_SectorBoardCell]:
    clamped_max_t = max(0, int(max_t))
    cols_n = max(2, int(cols))
    rows_n = max(2, int(rows))
    bucket_count = cols_n * rows_n
    t_now = int(timestep)
    t_target = int(t_star)
    view_start = int(start_t)
    view_end = int(end_t)
    objective_start = int(window_start)
    objective_end = int(window_end)
    normalized_focus: list[int] = []
    if focus_timesteps is not None:
        for raw in focus_timesteps:
            try:
                token = int(raw)
            except (TypeError, ValueError):
                continue
            if token not in normalized_focus:
                normalized_focus.append(token)
    elif focus_timestep is not None:
        normalized_focus = [int(focus_timestep)]
    edits_by_bucket = _bucketize_history_counts(
        history_counts=history_counts,
        max_t=clamped_max_t,
        bucket_count=bucket_count,
    )
    pressure_by_bucket = _bucketize_pressure_levels(
        pressure_levels=pressure_levels,
        max_t=clamped_max_t,
        bucket_count=bucket_count,
    )
    cells: list[_SectorBoardCell] = []
    for idx in range(bucket_count):
        row = idx // cols_n
        col = idx % cols_n
        bucket_start, bucket_end = _sector_bucket_bounds(
            index=idx,
            bucket_count=bucket_count,
            max_t=clamped_max_t,
        )
        if bucket_end < bucket_start:
            bucket_end = bucket_start

        marker = _bucket_marker(bucket_start, bucket_end, t_now=t_now, t_target=t_target)
        in_viewport = _ranges_overlap(bucket_start, bucket_end, view_start, view_end)
        in_objective_window = _ranges_overlap(
            bucket_start, bucket_end, objective_start, objective_end
        )
        edits = edits_by_bucket[idx]
        pressure_level = pressure_by_bucket[idx]
        focus_age: int | None = None
        for age, focus_t in enumerate(normalized_focus):
            if int(bucket_start) <= int(focus_t) <= int(bucket_end):
                focus_age = age
                break
        is_command_focus = focus_age is not None

        cells.append(
            _SectorBoardCell(
                row=row,
                col=col,
                start_t=bucket_start,
                end_t=bucket_end,
                pressure_level=pressure_level,
                edits=edits,
                marker=marker,
                in_viewport=in_viewport,
                in_objective_window=in_objective_window,
                is_command_focus=is_command_focus,
                focus_age=focus_age,
            )
        )
    return cells


def _sector_board_hover_summary(cell: _SectorBoardCell | None) -> str:
    if cell is None:
        return "Hover a board cell for sector-range details."
    cell_name = _sector_board_cell_name(row=cell.row, col=cell.col)
    marker_part = "marker=." if not cell.marker else f"marker={cell.marker}"
    if cell.focus_age is None:
        focus_part = "focus=."
    else:
        focus_part = f"focus=F{int(cell.focus_age)}"
    return (
        f"{cell_name} t={cell.start_t}..{cell.end_t} | "
        f"{_pressure_token(cell.pressure_level)} | "
        f"{_edits_token(cell.edits)} | {marker_part} | {focus_part}"
    )


def _sector_board_col_label(col: int) -> str:
    col_idx = max(0, int(col))
    # Spreadsheet-style labels scale naturally if the board grows past 26 cols.
    token: list[str] = []
    value = col_idx
    while True:
        value, remainder = divmod(value, 26)
        token.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(token))


def _sector_board_cell_name(*, row: int, col: int) -> str:
    return f"{_sector_board_col_label(col)}{max(1, int(row) + 1)}"


def _sector_board_cell_glyph(cell: _SectorBoardCell) -> str:
    if cell.marker:
        return cell.marker
    if cell.edits > 0 and cell.pressure_level is not None:
        return "*"
    if cell.edits > 0:
        return "e"
    if cell.pressure_level is not None:
        return "p"
    return "."


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
