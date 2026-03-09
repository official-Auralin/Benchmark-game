"""
Command-line entry points for generating, evaluating, and reporting GF-01 runs.

This module preserves the public CLI surface while delegating parser
registration and command execution to grouped submodules.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

import argparse

from .cli_registry import (
    register_p0_commands,
    register_pilot_commands,
    register_playback_commands,
    register_q033_commands,
    register_quality_commands,
    register_workflow_commands,
)
from .commands import (
    p0 as p0_commands,
    pilot as pilot_commands,
    playback as playback_commands,
    q033 as q033_commands,
    quality as quality_commands,
    shared as shared_commands,
    workflows as workflow_commands,
)
from .meta import DEFAULT_SPLIT_RATIOS, OFFICIAL_SPLITS
from .play import run_episode


_compute_manifest_coverage = shared_commands.compute_manifest_coverage
_validate_runs_manifest = shared_commands.validate_runs_manifest
_track_tool_policy_message = shared_commands.track_tool_policy_message
_adaptation_policy_message = shared_commands.adaptation_policy_message
_canonical_baseline_agent_id = shared_commands.canonical_baseline_agent_id
_baseline_panel_policy_message = shared_commands.baseline_panel_policy_message
_renderer_policy_message = shared_commands.renderer_policy_message
_parse_split_ratio_arg = shared_commands.parse_split_ratio_arg
_split_policy_report = shared_commands.split_policy_report
_manifest_instance_sets_by_split = shared_commands.manifest_instance_sets_by_split
_release_rotation_report = shared_commands.release_rotation_report
_build_report_payload = shared_commands.build_report_payload
_panel_ids = shared_commands.panel_ids
_track_for_agent_id = shared_commands.track_for_agent_id

_cmd_demo = workflow_commands.cmd_demo
_cmd_generate = workflow_commands.cmd_generate
_cmd_evaluate = workflow_commands.cmd_evaluate
_cmd_report = workflow_commands.cmd_report
_cmd_validate = workflow_commands.cmd_validate
_cmd_migrate_runs = workflow_commands.cmd_migrate_runs
_cmd_manifest = workflow_commands.cmd_manifest

_cmd_checks = quality_commands.cmd_checks
_cmd_profile = quality_commands.cmd_profile
_cmd_gate = quality_commands.cmd_gate
_cmd_split_policy_check = quality_commands.cmd_split_policy_check
_cmd_release_governance_check = quality_commands.cmd_release_governance_check
_cmd_release_report_check = quality_commands.cmd_release_report_check
_cmd_identifiability_check = quality_commands.cmd_identifiability_check
_invoke_subcommand_silently = quality_commands.invoke_subcommand_silently
_cmd_release_candidate_check = quality_commands.cmd_release_candidate_check

_cmd_q033_build_manifests = q033_commands.cmd_q033_build_manifests
_cmd_q033_sweep = q033_commands.cmd_q033_sweep
_cmd_q033_closure_check = q033_commands.cmd_q033_closure_check

_parse_seed_list = pilot_commands.parse_seed_list
_cmd_freeze_pilot = pilot_commands.cmd_freeze_pilot
_cmd_pilot_campaign = pilot_commands.cmd_pilot_campaign
_load_external_episode_payloads = pilot_commands.load_external_episode_payloads
_sha256_file = pilot_commands.sha256_file
_cmd_release_package = pilot_commands.cmd_release_package
_analysis_rate = pilot_commands.analysis_rate
_complexity_values = pilot_commands.complexity_values
_complexity_score = pilot_commands.complexity_score
_assign_numeric_quartiles = pilot_commands.assign_numeric_quartiles
_assign_complexity_quartiles = pilot_commands.assign_complexity_quartiles
_quartile_stats = pilot_commands.quartile_stats
_pearson_corr = pilot_commands.pearson_corr
_complexity_knob_diagnostics = pilot_commands.complexity_knob_diagnostics
_agent_summary_rows = pilot_commands.agent_summary_rows
_cmd_pilot_analyze = pilot_commands.cmd_pilot_analyze

_load_feedback_rows = p0_commands.load_feedback_rows
_cmd_p0_feedback_check = p0_commands.cmd_p0_feedback_check
_cmd_p0_feedback_template = p0_commands.cmd_p0_feedback_template
_parse_seed_list_flexible = p0_commands.parse_seed_list_flexible
_cmd_p0_session_check = p0_commands.cmd_p0_session_check
_cmd_p0_gate = p0_commands.cmd_p0_gate
_cmd_p0_seed_pack = p0_commands.cmd_p0_seed_pack
_cmd_p0_init = p0_commands.cmd_p0_init


def _cmd_play(args: argparse.Namespace) -> int:
    """Compatibility wrapper that preserves `gf01.cli.run_episode` patch points."""
    original_run_episode = playback_commands.run_episode
    playback_commands.run_episode = run_episode
    try:
        return int(playback_commands.cmd_play(args))
    finally:
        playback_commands.run_episode = original_run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GF-01 benchmark harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    default_ratio_arg = ",".join(
        f"{split_id}={DEFAULT_SPLIT_RATIOS[split_id]}" for split_id in OFFICIAL_SPLITS
    )

    register_workflow_commands(sub)
    register_quality_commands(sub, default_ratio_arg=default_ratio_arg)
    register_q033_commands(sub)
    register_pilot_commands(sub)
    register_playback_commands(sub)
    register_p0_commands(sub)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
