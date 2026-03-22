"""P0/internal-alpha command parser registration."""

from __future__ import annotations

import argparse

from .commands.p0 import (
    cmd_p0_feedback_check,
    cmd_p0_feedback_template,
    cmd_p0_gate,
    cmd_p0_init,
    cmd_p0_seed_pack,
    cmd_p0_session_check,
)
from .meta import ALLOWED_MODES, ALLOWED_RENDERER_TRACKS


def register_p0_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_p0 = sub.add_parser(
        "p0-feedback-check",
        help="Evaluate internal-alpha (P0) feedback CSV against clarity/blocker thresholds",
    )
    p_p0.add_argument("--feedback", type=str, required=True, help="CSV with required columns: tester_id, objective_clarity, control_clarity, action_effect_clarity, must_fix_blockers")
    p_p0.add_argument("--min-score", type=int, default=3, help="Minimum per-tester clarity score counted as acceptable")
    p_p0.add_argument("--min-ratio", type=float, default=0.80, help="Minimum acceptable ratio for each clarity metric")
    p_p0.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_p0.set_defaults(func=cmd_p0_feedback_check)

    p_p0_session = sub.add_parser(
        "p0-session-check",
        help="Verify P0 play-session artifacts exist and match feedback seed/backend declarations",
    )
    p_p0_session.add_argument("--feedback", type=str, required=True, help="Feedback CSV (must include tester_id, backend_used, seed_list_run)")
    p_p0_session.add_argument("--runs-dir", type=str, default="p0_runs", help="Directory containing per-session play artifacts named <tester_id>_<seed>.json")
    p_p0_session.add_argument("--required-renderer-track", type=str, default="visual", choices=list(ALLOWED_RENDERER_TRACKS), help="Expected renderer_track in each play artifact run_contract")
    p_p0_session.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_p0_session.set_defaults(func=cmd_p0_session_check)

    p_p0_gate = sub.add_parser(
        "p0-gate",
        help="Run P0 session-coverage and feedback-threshold checks in sequence and emit a single pass/fail summary",
    )
    p_p0_gate.add_argument("--feedback", type=str, required=True, help="Feedback CSV used for both session and feedback checks")
    p_p0_gate.add_argument("--runs-dir", type=str, default="p0_runs", help="Directory containing per-session play artifacts named <tester_id>_<seed>.json")
    p_p0_gate.add_argument("--required-renderer-track", type=str, default="visual", choices=list(ALLOWED_RENDERER_TRACKS), help="Expected renderer_track in each play artifact run_contract")
    p_p0_gate.add_argument("--min-score", type=int, default=3, help="Minimum per-tester clarity score counted as acceptable")
    p_p0_gate.add_argument("--min-ratio", type=float, default=0.80, help="Minimum acceptable ratio for each clarity metric")
    p_p0_gate.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_p0_gate.set_defaults(func=cmd_p0_gate)

    p_p0_template = sub.add_parser(
        "p0-feedback-template",
        help="Write a deterministic CSV template for P0 internal-alpha feedback collection",
    )
    p_p0_template.add_argument("--out", type=str, default="p0_feedback_template.csv", help="Path to write feedback CSV template")
    p_p0_template.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
    p_p0_template.set_defaults(func=cmd_p0_feedback_template)

    p_p0_seed = sub.add_parser(
        "p0-seed-pack",
        help="Freeze a deterministic P0 internal-alpha seed pack using canonical defaults",
    )
    p_p0_seed.add_argument("--freeze-id", type=str, default="gf01-p0-alpha-v1")
    p_p0_seed.add_argument("--split", type=str, default="pilot_internal_p0_v1")
    p_p0_seed.add_argument("--seed-start", type=int, default=7000)
    p_p0_seed.add_argument("--count", type=int, default=8)
    p_p0_seed.add_argument("--seeds", type=str, default="", help="Optional explicit comma-separated seed list; overrides --seed-start/--count")
    p_p0_seed.add_argument("--mode", type=str, default="normal", choices=list(ALLOWED_MODES), help="P0 default is normal mode; override only for targeted checks")
    p_p0_seed.add_argument("--out-dir", type=str, default="pilot_freeze/gf01_p0_alpha_v1", help="Output directory for the frozen P0 pack")
    p_p0_seed.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory")
    p_p0_seed.set_defaults(func=cmd_p0_seed_pack)

    p_p0_init = sub.add_parser(
        "p0-init",
        help="One-shot P0 setup: feedback template plus deterministic seed pack",
    )
    p_p0_init.add_argument("--template-out", type=str, default="p0_feedback.csv", help="Path to write P0 feedback CSV template")
    p_p0_init.add_argument("--freeze-id", type=str, default="gf01-p0-alpha-v1")
    p_p0_init.add_argument("--split", type=str, default="pilot_internal_p0_v1")
    p_p0_init.add_argument("--seed-start", type=int, default=7000)
    p_p0_init.add_argument("--count", type=int, default=8)
    p_p0_init.add_argument("--seeds", type=str, default="", help="Optional explicit comma-separated seed list; overrides --seed-start/--count")
    p_p0_init.add_argument("--mode", type=str, default="normal", choices=list(ALLOWED_MODES), help="P0 default is normal mode; override only for targeted checks")
    p_p0_init.add_argument("--out-dir", type=str, default="pilot_freeze/gf01_p0_alpha_v1", help="Output directory for the frozen P0 pack")
    p_p0_init.add_argument("--force", action="store_true", help="Overwrite template output and non-empty P0 seed-pack directory")
    p_p0_init.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_p0_init.set_defaults(func=cmd_p0_init)
