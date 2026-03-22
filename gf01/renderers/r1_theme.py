"""Deterministic mapping from formal GF-01 concepts to a causal-board vocabulary.

This module is pure logic with no pygame dependency. It keeps labels and colors
grounded in the actual AP identities rather than an arbitrary flavor layer.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GF01Instance

_TILE_PALETTE = {
    "neutral":   (64, 81, 96),
    "input":     (46, 104, 92),
    "output":    (52, 90, 132),
    "objective": (154, 114, 52),
    "highlight": (88, 122, 88),
    "bridge":    (92, 82, 68),
}

COLOR_BG         = (14, 18, 26)
COLOR_PANEL      = (22, 28, 40)
COLOR_PANEL_LITE = (30, 38, 54)
COLOR_BORDER     = (58, 72, 96)
COLOR_BORDER_HI  = (92, 118, 156)
COLOR_TEXT        = (210, 222, 240)
COLOR_TEXT_DIM    = (148, 162, 186)
COLOR_TEXT_BRIGHT = (240, 248, 255)
COLOR_ACCENT_G   = (72, 186, 112)
COLOR_ACCENT_R   = (196, 72, 64)
COLOR_ACCENT_Y   = (212, 188, 96)
COLOR_ACCENT_C   = (72, 172, 212)
COLOR_ACCENT_P   = (154, 132, 92)
COLOR_ENERGY     = (96, 192, 144)
COLOR_CMD_PTS    = (112, 168, 216)
COLOR_DEPLOY_BTN = (52, 148, 92)
COLOR_SKIP_BTN   = (68, 78, 98)
COLOR_UNDO_BTN   = (148, 62, 56)
COLOR_VICTORY    = (48, 164, 96)
COLOR_DEFEAT     = (172, 56, 48)
COLOR_FLASH_G    = (62, 196, 112)
COLOR_FLASH_R    = (212, 68, 56)
COLOR_STAGED_ON  = (56, 186, 108)
COLOR_STAGED_OFF = (186, 56, 48)
COLOR_HISTORY_ON = (48, 148, 92)
COLOR_HISTORY_OFF = (148, 48, 42)


def _humanize_ap(ap: str) -> str:
    text = str(ap).replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Unnamed AP"
    return " ".join(part.capitalize() for part in text.split())


def _stable_index(key: str, modulo: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, modulo)


def defense_name(ap: str, input_aps: list[str]) -> str:
    return _humanize_ap(ap)


def defense_icon(ap: str, input_aps: list[str]) -> str:
    _ = input_aps
    return "\u25A0"


def defense_color(ap: str, input_aps: list[str]) -> tuple[int, int, int]:
    try:
        idx = input_aps.index(ap)
    except ValueError:
        idx = _stable_index(ap, 8)
    palette = (
        (72, 170, 132),
        (86, 154, 200),
        (196, 156, 88),
        (116, 152, 86),
        (88, 180, 172),
        (212, 132, 96),
        (156, 184, 112),
        (184, 120, 128),
    )
    return palette[idx % len(palette)]


def threat_name(output_ap: str, output_aps: list[str]) -> str:
    _ = output_aps
    return _humanize_ap(output_ap)


def wave_label(timestep: int, total: int) -> str:
    return f"STEP {timestep + 1} / {total}"


def critical_wave_label(t_star: int) -> str:
    return f"TARGET STEP: {t_star + 1}"


def energy_label(budget_t_remaining: int) -> str:
    return f"STEP BUDGET: {budget_t_remaining}"


def cmd_points_label(budget_a_remaining: int) -> str:
    return f"LEGACY ATOMS: {budget_a_remaining}"


def objective_text_themed(instance: "GF01Instance") -> str:
    target_name = threat_name(instance.effect_ap, instance.automaton.output_aps)
    wave_num = instance.t_star + 1
    if instance.mode == "hard":
        return f"Cause {target_name}=1 at exactly step {wave_num}"
    start = max(0, instance.t_star - instance.window_size) + 1
    if start == wave_num:
        return f"Cause {target_name}=1 at step {wave_num}"
    return f"Cause {target_name}=1 during steps {start}-{wave_num}"


def effect_status_display(status: str) -> tuple[str, tuple[int, int, int]]:
    token = str(status).strip().lower()
    if token == "triggered":
        return ("TARGET TRIGGERED", COLOR_ACCENT_G)
    if token == "not-triggered":
        return ("TARGET NOT TRIGGERED", COLOR_ACCENT_R)
    return ("OBSERVING", COLOR_BORDER)


def tile_type_for_index(idx: int, total: int, seed: int) -> str:
    _ = (idx, total, seed)
    return "neutral"


def tile_color(tile_type: str) -> tuple[int, int, int]:
    return _TILE_PALETTE.get(tile_type, _TILE_PALETTE["neutral"])


def victory_text(goal: bool, suff: bool, min1: bool) -> tuple[str, str, tuple[int, int, int]]:
    if goal and suff and min1:
        return ("CERTIFIED", "Target achieved with singleton-minimal certificate.", COLOR_VICTORY)
    if goal and suff:
        return ("GOAL MET", "Target achieved, but intervention minimality can improve.", COLOR_VICTORY)
    if goal:
        return ("PARTIAL", "Target occurred, but certificate validity is incomplete.", COLOR_ACCENT_Y)
    return ("GOAL MISSED", "The target effect never occurred in the allowed region.", COLOR_DEFEAT)
