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

from dataclasses import dataclass
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GF01Instance


_SESSION: "_R1PygameSession | None" = None


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


def _help_overlay_lines() -> list[str]:
    return [
        "GF-01-R1 quick help",
        "Goal: commit interventions to reach the visible objective.",
        "Mouse: click 0/1 to set AP, clear to unset AP.",
        "Keys: 1..9,0 cycle AP slots on current page.",
        "Keys: Left/Right page APs | +/- AP density | G AP group filter.",
        "Keys: Enter commit | Esc skip | Backspace clear all.",
        "Tip: use Output delta to see what changed after each step.",
        "Press H to hide/show this panel.",
    ]


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


def _timeline_mark(t: int, timestep: int, t_star: int) -> str:
    if t == timestep and t == t_star:
        return "B"
    if t == timestep:
        return "N"
    if t == t_star:
        return "T"
    return ""


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
        self._show_help_overlay = True

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

    def _draw_timeline(
        self,
        *,
        timestep: int,
        t_star: int,
        history_atoms: object,
    ) -> None:
        history_counts: dict[int, int] = {}
        if isinstance(history_atoms, list):
            for item in history_atoms:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    continue
                try:
                    t_item = int(item[0])
                except (TypeError, ValueError):
                    continue
                history_counts[t_item] = history_counts.get(t_item, 0) + 1

        max_t = max([0, timestep, t_star, *history_counts.keys()])
        cols = min(max_t + 1, 32)
        cell_w = 26
        x0 = 24
        y0 = 130
        self._draw_text("Timeline sectors (t):", x0, y0 - 26)
        for t in range(cols):
            x = x0 + t * (cell_w + 3)
            fill = (34, 44, 62)
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
            mark = _timeline_mark(t, timestep, t_star)
            if mark:
                self._draw_text(mark, x + 9, y0 - 16, small=True, color=(196, 212, 236))
            self._draw_text(str(t), x + 7, y0 + 8, small=True)
            edits = history_counts.get(t)
            if edits is not None:
                self._draw_text(f"{edits}", x + 9, y0 + 38, small=True)
        self._draw_text("marks: N=now, T=target, B=both", x0, y0 + 58, small=True)
        self._draw_text("edits per t shown below sectors", x0, y0 + 74, small=True)

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

    def _draw_controls(
        self,
        *,
        input_aps: list[str],
        pending: dict[str, int],
        y_start: int,
        page: int,
        page_size: int,
        group_filter: str | None,
    ) -> tuple[list[_Button], int, int, int]:
        buttons: list[_Button] = []
        self._draw_text("Set interventions for current timestep:", 24, y_start - 26)
        visible_aps, page, total_pages = _paginate_input_aps(
            input_aps,
            page=page,
            page_size=page_size,
        )
        group_label = "ALL" if group_filter is None else group_filter
        self._draw_text(
            f"AP page {page + 1}/{total_pages} | page size={page_size}  "
            f"group={group_label} (Left/Right page, +/- density, G group)",
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
    ) -> dict[str, int]:
        pending: dict[str, int] = {}
        current_buttons: list[_Button] = []
        page = 0
        page_size = 10
        input_aps_all = list(instance.automaton.input_aps)
        group_keys = list(_grouped_input_aps(input_aps_all).keys())
        group_filter: str | None = None
        if timestep == 0:
            self._previous_observed_y_t = None
            self._show_help_overlay = True
        current_y_t = _normalize_binary_map(
            None if last_obs is None else last_obs.get("y_t", {})
        )
        delta_summary = _describe_output_delta(self._previous_observed_y_t, current_y_t)
        if current_y_t:
            self._previous_observed_y_t = dict(current_y_t)
        while True:
            visible_pool = _apply_group_filter(input_aps_all, group_filter)
            visible_aps, page, _ = _paginate_input_aps(
                visible_pool,
                page=page,
                page_size=page_size,
            )
            self.clock.tick(30)
            for event in self.pg.event.get():
                if event.type == self.pg.QUIT:  # pragma: no cover - UI event
                    return {}
                if event.type == self.pg.KEYDOWN:
                    if event.key in (self.pg.K_RETURN, self.pg.K_KP_ENTER):
                        return dict(pending)
                    if event.key == self.pg.K_ESCAPE:
                        return {}
                    if event.key == self.pg.K_BACKSPACE:
                        pending.clear()
                    if event.key == self.pg.K_h:
                        self._show_help_overlay = not self._show_help_overlay
                    if event.key in (self.pg.K_LEFT, self.pg.K_PAGEUP):
                        page = max(0, page - 1)
                    if event.key in (self.pg.K_RIGHT, self.pg.K_PAGEDOWN):
                        # Clamp later using pagination helper.
                        page += 1
                    if event.key in (self.pg.K_EQUALS, self.pg.K_PLUS, self.pg.K_KP_PLUS):
                        page_size = _clamp_page_size(page_size + 1)
                    if event.key in (self.pg.K_MINUS, self.pg.K_UNDERSCORE, self.pg.K_KP_MINUS):
                        page_size = _clamp_page_size(page_size - 1)
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

            history_atoms = [] if last_obs is None else last_obs.get("history_atoms", [])
            self._draw_timeline(
                timestep=timestep,
                t_star=int(instance.t_star),
                history_atoms=history_atoms,
            )
            current_buttons, page, _, _ = self._draw_controls(
                input_aps=visible_pool,
                pending=pending,
                y_start=250,
                page=page,
                page_size=page_size,
                group_filter=group_filter,
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
                self._draw_text("Observation Summary:", status_x, status_y)
                self._draw_text(
                    _summarize_observed_outputs(y_t), status_x, status_y + 28
                )
                self._draw_text(
                    f"Effect status: {effect} | Budget remaining: timesteps={bt}, atoms={ba}",
                    status_x,
                    status_y + 56,
                )
                self._draw_text(delta_summary, status_x, status_y + 84)
                self._draw_text(
                    _summarize_pending_interventions(pending),
                    status_x,
                    status_y + 112,
                    small=True,
                    color=(192, 209, 232),
                )

            footer_y = 712
            self._draw_text("Controls:", 24, footer_y, small=True)
            self._draw_text(
                "Click 0/1 to set AP value, clear to unset | 1..9,0 cycle APs | "
                "Enter=commit | Esc=skip | Backspace=clear all | "
                "Left/Right=AP page | +/-=AP density | H=help",
                100,
                footer_y,
                small=True,
            )
            if self._show_help_overlay:
                self._draw_help_overlay()
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
) -> dict[str, int]:
    return _session().choose_action(
        last_obs=last_obs,
        timestep=timestep,
        instance=instance,
        objective_text=objective_text,
    )
