"""Playback command parser registration."""

from __future__ import annotations

import argparse

from .commands.playback import cmd_play
from .meta import (
    ALLOWED_ADAPTATION_CONDITIONS,
    ALLOWED_ADAPTATION_DATA_SCOPES,
    ALLOWED_EVAL_TRACKS,
    ALLOWED_RENDERER_TRACKS,
)


def register_playback_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_play = sub.add_parser(
        "play",
        help="Run one playable GF-01 episode (human, scripted, or baseline agent)",
    )
    p_play.add_argument("--seed", type=int, default=1337)
    p_play.add_argument("--split", type=str, default="public_dev")
    p_play.add_argument("--instances", type=str, default="", help="Optional instance bundle/list JSON")
    p_play.add_argument("--instance-index", type=int, default=0, help="Instance index when --instances contains multiple entries")
    p_play.add_argument("--agent", type=str, default="", help="Optional baseline policy id")
    p_play.add_argument("--script", type=str, default="", help="Optional action script JSON")
    p_play.add_argument("--renderer-track", type=str, choices=list(ALLOWED_RENDERER_TRACKS), default="visual")
    p_play.add_argument(
        "--visual-backend",
        type=str,
        default="text",
        choices=["text", "pygame"],
        help="Human-play visual backend. 'text' uses terminal snapshots; 'pygame' opens the canonical GF-01-R1 tower-defense graphical window.",
    )
    p_play.add_argument("--eval-track", type=str, default="EVAL-CB", choices=list(ALLOWED_EVAL_TRACKS))
    p_play.add_argument("--tool-allowlist-id", type=str, default="none")
    p_play.add_argument("--tool-log-hash", type=str, default="")
    p_play.add_argument(
        "--adaptation-condition",
        type=str,
        default="no_adaptation",
        choices=list(ALLOWED_ADAPTATION_CONDITIONS),
    )
    p_play.add_argument("--adaptation-budget-tokens", type=int, default=0)
    p_play.add_argument(
        "--adaptation-data-scope",
        type=str,
        default="none",
        choices=list(ALLOWED_ADAPTATION_DATA_SCOPES),
    )
    p_play.add_argument("--adaptation-protocol-id", type=str, default="none")
    p_play.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_play.set_defaults(func=cmd_play)
