"""Deterministic theme mapping from formal GF-01 concepts to tower-defense game vocabulary.

This module is pure logic with no pygame dependency. It translates APs,
timesteps, budgets, and effect status into player-facing game terms so that
the visual renderer never exposes formal notation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GF01Instance

_DEFENSE_NAMES: tuple[str, ...] = (
    "Shield Gen",
    "Pulse Cannon",
    "Barrier Wall",
    "Flux Emitter",
    "Phase Gate",
    "Arc Pylon",
    "Cryo Field",
    "Grav Anchor",
    "Ion Turret",
    "Rift Beacon",
    "Nano Swarm",
    "Mag Lance",
    "Thorn Mine",
    "Void Trap",
    "Spark Relay",
    "Storm Array",
)

_THREAT_NAMES: tuple[str, ...] = (
    "Sector Alpha",
    "Sector Beta",
    "Sector Gamma",
    "Sector Delta",
    "Sector Epsilon",
    "Sector Zeta",
    "Sector Eta",
    "Sector Theta",
)

_DEFENSE_ICONS: tuple[str, ...] = (
    "\u25C8",  # ◈ diamond
    "\u2726",  # ✦ star
    "\u2588",  # █ block
    "\u2741",  # ❁ flower
    "\u25CE",  # ◎ bullseye
    "\u2607",  # ☇ lightning
    "\u2746",  # ❆ snowflake
    "\u2693",  # ⚓ anchor
    "\u25B2",  # ▲ triangle
    "\u25C6",  # ◆ filled diamond
    "\u2742",  # ❂ circled star
    "\u2694",  # ⚔ swords
    "\u2620",  # ☠ skull
    "\u2B25",  # ⬥ medium diamond
    "\u26A1",  # ⚡ zap
    "\u2604",  # ☄ comet
)

_TILE_PALETTE = {
    "ground":    (58, 104, 72),
    "water":     (38, 82, 132),
    "rock":      (88, 78, 68),
    "objective": (118, 102, 46),
    "highlight": (68, 128, 96),
    "danger":    (128, 52, 44),
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
COLOR_ACCENT_P   = (158, 112, 196)
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


def defense_name(ap: str, input_aps: list[str]) -> str:
    try:
        idx = input_aps.index(ap)
    except ValueError:
        idx = abs(hash(ap))
    return _DEFENSE_NAMES[idx % len(_DEFENSE_NAMES)]


def defense_icon(ap: str, input_aps: list[str]) -> str:
    try:
        idx = input_aps.index(ap)
    except ValueError:
        idx = abs(hash(ap))
    return _DEFENSE_ICONS[idx % len(_DEFENSE_ICONS)]


def defense_color(ap: str, input_aps: list[str]) -> tuple[int, int, int]:
    try:
        idx = input_aps.index(ap)
    except ValueError:
        idx = abs(hash(ap))
    palette = (
        (72, 186, 132),   # teal-green
        (132, 168, 216),  # steel-blue
        (196, 156, 88),   # amber
        (168, 112, 196),  # purple
        (96, 192, 172),   # cyan
        (212, 132, 96),   # copper
        (148, 196, 108),  # lime
        (196, 112, 148),  # rose
    )
    return palette[idx % len(palette)]


def threat_name(output_ap: str, output_aps: list[str]) -> str:
    try:
        idx = output_aps.index(output_ap)
    except ValueError:
        idx = abs(hash(output_ap))
    return _THREAT_NAMES[idx % len(_THREAT_NAMES)]


def wave_label(timestep: int, total: int) -> str:
    return f"WAVE {timestep + 1} / {total}"


def critical_wave_label(t_star: int) -> str:
    return f"CRITICAL WAVE: {t_star + 1}"


def energy_label(budget_t_remaining: int) -> str:
    return f"ENERGY: {budget_t_remaining}"


def cmd_points_label(budget_a_remaining: int) -> str:
    return f"COMMANDS: {budget_a_remaining}"


def objective_text_themed(instance: "GF01Instance") -> str:
    target_name = threat_name(instance.effect_ap, instance.automaton.output_aps)
    wave_num = instance.t_star + 1
    if instance.mode == "hard":
        return f"Defend {target_name} on exactly Wave {wave_num}"
    start = max(0, instance.t_star - instance.window_size) + 1
    if start == wave_num:
        return f"Defend {target_name} on Wave {wave_num}"
    return f"Defend {target_name} during Waves {start}\u2013{wave_num}"


def effect_status_display(status: str) -> tuple[str, tuple[int, int, int]]:
    token = str(status).strip().lower()
    if token == "triggered":
        return ("BASE SECURE", COLOR_ACCENT_G)
    if token == "not-triggered":
        return ("UNDER THREAT", COLOR_ACCENT_R)
    return ("SCANNING...", COLOR_BORDER)


def tile_type_for_index(idx: int, total: int, seed: int) -> str:
    h = (idx * 2654435761 + seed) & 0xFFFFFFFF
    frac = (h % 100) / 100.0
    if frac < 0.60:
        return "ground"
    if frac < 0.78:
        return "rock"
    if frac < 0.90:
        return "water"
    return "ground"


def tile_color(tile_type: str) -> tuple[int, int, int]:
    return _TILE_PALETTE.get(tile_type, _TILE_PALETTE["ground"])


def victory_text(goal: bool, suff: bool, min1: bool) -> tuple[str, str, tuple[int, int, int]]:
    if goal and suff and min1:
        return ("VICTORY", "Base defended with optimal deployment!", COLOR_VICTORY)
    if goal and suff:
        return ("VICTORY", "Base defended! Deploy efficiency could improve.", COLOR_VICTORY)
    if goal:
        return ("SURVIVED", "Base held, but defenses were not optimal.", COLOR_ACCENT_Y)
    return ("DEFEAT", "The base has been overrun.", COLOR_DEFEAT)
