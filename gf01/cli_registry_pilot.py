"""Pilot workflow command parser registration."""

from __future__ import annotations

import argparse

from .commands.pilot import (
    cmd_freeze_pilot,
    cmd_pilot_analyze,
    cmd_pilot_campaign,
    cmd_release_package,
)
from .meta import (
    ALLOWED_ADAPTATION_CONDITIONS,
    ALLOWED_ADAPTATION_DATA_SCOPES,
    ALLOWED_BASELINE_PANEL_LEVELS,
    ALLOWED_MODES,
    ALLOWED_RENDERER_TRACKS,
    ALLOWED_TOOL_ALLOWLISTS_BY_TRACK,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
)


def register_pilot_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_freeze = sub.add_parser(
        "freeze-pilot",
        help="Freeze a provisional internal pilot pack (bundle + manifest + freeze metadata)",
    )
    p_freeze.add_argument("--freeze-id", type=str, default="gf01-pilot-freeze-v1")
    p_freeze.add_argument("--split", type=str, default="pilot_internal_v1")
    p_freeze.add_argument("--seed-start", type=int, default=7000)
    p_freeze.add_argument("--count", type=int, default=24)
    p_freeze.add_argument("--seeds", type=str, default="", help="Optional comma-separated explicit seed list; overrides --seed-start/--count")
    p_freeze.add_argument(
        "--mode",
        type=str,
        default="",
        help="Optional mode override for all frozen instances (normal/hard). Leave empty for mixed generation.",
    )
    p_freeze.add_argument("--out-dir", type=str, default="pilot_freeze/gf01_pilot_freeze_v1")
    p_freeze.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory")
    p_freeze.set_defaults(func=cmd_freeze_pilot)

    p_campaign = sub.add_parser(
        "pilot-campaign",
        help="Run a pilot campaign on a frozen pack with official validation/report outputs",
    )
    p_campaign.add_argument("--freeze-dir", type=str, required=True, help="Directory containing instance_bundle_v1.json and split_manifest_v1.json")
    p_campaign.add_argument("--out-dir", type=str, default="pilot_runs/gf01_pilot_campaign_v1", help="Output directory for combined runs, validation, and report artifacts")
    p_campaign.add_argument("--baseline-panel", type=str, default="random,greedy,search,tool,oracle", help="Comma-separated baseline ids (e.g., random,greedy,search,tool,oracle)")
    p_campaign.add_argument(
        "--baseline-policy-level",
        type=str,
        default="full",
        choices=list(ALLOWED_BASELINE_PANEL_LEVELS),
        help="'full' requires random,greedy,search,tool,oracle; 'core' requires random,greedy,oracle.",
    )
    p_campaign.add_argument("--renderer-track", type=str, default="json", choices=list(ALLOWED_RENDERER_TRACKS))
    p_campaign.add_argument("--seed", type=int, default=1100)
    p_campaign.add_argument(
        "--tool-allowlist-id",
        type=str,
        default=DEFAULT_TOOL_ALLOWLIST_BY_TRACK["EVAL-TA"],
        choices=list(ALLOWED_TOOL_ALLOWLISTS_BY_TRACK["EVAL-TA"]),
        help="Tool allowlist id used for EVAL-TA baseline rows",
    )
    p_campaign.add_argument("--tool-log-hash", type=str, default="")
    p_campaign.add_argument("--external-runs", action="append", default=[], help="Optional path to external run JSONL (repeatable)")
    p_campaign.add_argument("--external-episodes", action="append", default=[], help="Optional play-output artifact path (JSON or JSONL). Each payload is converted to strict run rows and merged.")
    p_campaign.add_argument(
        "--adaptation-condition",
        type=str,
        default="no_adaptation",
        choices=list(ALLOWED_ADAPTATION_CONDITIONS),
    )
    p_campaign.add_argument("--adaptation-budget-tokens", type=int, default=0)
    p_campaign.add_argument(
        "--adaptation-data-scope",
        type=str,
        default="none",
        choices=list(ALLOWED_ADAPTATION_DATA_SCOPES),
    )
    p_campaign.add_argument("--adaptation-protocol-id", type=str, default="none")
    p_campaign.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory")
    p_campaign.set_defaults(func=cmd_pilot_campaign)

    p_release_pkg = sub.add_parser(
        "release-package",
        help="Build a reproducibility package from a frozen pack and campaign artifacts with strict validation.",
    )
    p_release_pkg.add_argument("--freeze-dir", type=str, required=True, help="Directory containing frozen pilot artifacts (bundle + manifest)")
    p_release_pkg.add_argument("--campaign-dir", type=str, required=True, help="Directory containing campaign artifacts (runs + validation + report)")
    p_release_pkg.add_argument("--out-dir", type=str, default="release_packages/gf01_release_package_v1", help="Output directory for packaged artifacts and manifest")
    p_release_pkg.add_argument("--force", action="store_true", help="Overwrite an existing non-empty output directory")
    p_release_pkg.set_defaults(func=cmd_release_package)

    p_analysis = sub.add_parser(
        "pilot-analyze",
        help="Analyze a pilot campaign and evaluate DEC-014d calibration triggers",
    )
    p_analysis.add_argument("--campaign-dir", type=str, required=True, help="Directory containing campaign artifacts (runs_combined.jsonl at minimum)")
    p_analysis.add_argument("--out", type=str, default="", help="Optional output path for analysis JSON (default: <campaign-dir>/pilot_analysis.json)")
    p_analysis.add_argument("--eval-track", type=str, default="EVAL-CB")
    p_analysis.add_argument("--mode", type=str, default="normal", choices=list(ALLOWED_MODES))
    p_analysis.add_argument("--greedy-agent-name", type=str, default="BL-01-GreedyLocal")
    p_analysis.add_argument("--public-splits", type=str, default="public_dev,public_val", help="Comma-separated split IDs treated as public (excluded from held-out shortcut check)")
    p_analysis.add_argument("--sample-target", type=int, default=240)
    p_analysis.add_argument("--discrimination-delta-threshold", type=float, default=0.12)
    p_analysis.add_argument("--discrimination-q4-floor", type=float, default=0.10)
    p_analysis.add_argument("--shortcut-goal-threshold", type=float, default=0.40)
    p_analysis.add_argument("--shortcut-certified-floor", type=float, default=0.05)
    p_analysis.set_defaults(func=cmd_pilot_analyze)
