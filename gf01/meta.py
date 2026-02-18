"""
Shared metadata helpers for versioned GF-01 artifacts and reproducibility.

This module centralizes schema versions, benchmark/version labels, and stable
hash helpers so generated bundles, run logs, and manifests can be audited
consistently across commands and machines.
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

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any

FAMILY_ID = "GF-01"
BENCHMARK_VERSION = "0.1.0-dev"
GENERATOR_VERSION = "0.1.0-dev"
CHECKER_VERSION = "0.1.0-dev"
HARNESS_VERSION = __version__

INSTANCE_BUNDLE_SCHEMA_VERSION = "gf01.instance_bundle.v1"
RUN_RECORD_SCHEMA_VERSION = "gf01.run_record.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "gf01.split_manifest.v1"
PILOT_FREEZE_SCHEMA_VERSION = "gf01.pilot_freeze.v1"

ALLOWED_EVAL_TRACKS = ("EVAL-CB", "EVAL-TA", "EVAL-OC")
ALLOWED_MODES = ("normal", "hard")
ALLOWED_PLAY_PROTOCOLS = ("commit_only",)
ADAPTATION_POLICY_VERSION = "gf01.adaptation_policy.v1"
ALLOWED_ADAPTATION_CONDITIONS = (
    "no_adaptation",
    "prompt_adaptation",
    "weight_finetune",
)
ALLOWED_ADAPTATION_DATA_SCOPES = (
    "none",
    "public_only",
    "public_plus_external",
)
RENDERER_POLICY_VERSION = "gf01.renderer_policy.v1"
ALLOWED_RENDERER_TRACKS = ("json", "visual")
RENDERER_PROFILE_BY_TRACK = {
    "json": "canonical-json-v1",
    "visual": "GF-01-R1",
}
ALLOWED_RENDERER_PROFILE_IDS = tuple(sorted(RENDERER_PROFILE_BY_TRACK.values()))
BASELINE_PANEL_POLICY_VERSION = "gf01.baseline_panel_policy.v1"
ALLOWED_BASELINE_PANEL_LEVELS = ("full", "core")
BASELINE_PANEL_CORE = ("random", "greedy", "oracle")
BASELINE_PANEL_FULL = ("random", "greedy", "search", "tool", "oracle")
OFFICIAL_SPLITS = ("public_dev", "public_val", "private_eval")
SPLIT_POLICY_VERSION = "gf01.split_policy.v1"
DEFAULT_SPLIT_RATIOS = {
    "public_dev": 0.20,
    "public_val": 0.20,
    "private_eval": 0.60,
}
DEFAULT_SPLIT_RATIO_TOLERANCE = 0.05
DEFAULT_PRIVATE_EVAL_MIN_COUNT = 1
TOOL_POLICY_VERSION = "gf01.tool_policy.v1"
ALLOWED_TOOL_ALLOWLISTS_BY_TRACK = {
    "EVAL-CB": ("none",),
    "EVAL-TA": ("local-planner-v1",),
    "EVAL-OC": ("oracle-exact-search-v1",),
}
DEFAULT_TOOL_ALLOWLIST_BY_TRACK = {
    track: allowlists[0] for track, allowlists in ALLOWED_TOOL_ALLOWLISTS_BY_TRACK.items()
}

REQUIRED_RUN_FIELDS = (
    "schema_version",
    "family_id",
    "benchmark_version",
    "generator_version",
    "checker_version",
    "harness_version",
    "git_commit",
    "config_hash",
    "instance_id",
    "eval_track",
    "renderer_track",
    "renderer_policy_version",
    "renderer_profile_id",
    "agent_name",
    "certificate",
    "suff",
    "min1",
    "valid",
    "goal",
    "ap_f1",
    "ts_f1",
    "split_id",
    "mode",
    "tool_allowlist_id",
    "play_protocol",
    "scored_commit_episode",
    "adaptation_policy_version",
    "adaptation_condition",
    "adaptation_budget_tokens",
    "adaptation_data_scope",
    "adaptation_protocol_id",
)

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "family_id",
    "benchmark_version",
    "generator_version",
    "checker_version",
    "harness_version",
    "instance_count",
    "instances",
)


def stable_hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(cfg: Any) -> str:
    if is_dataclass(cfg):
        return stable_hash_json(asdict(cfg))
    return stable_hash_json(cfg)


def renderer_profile_for_track(renderer_track: str) -> str:
    return RENDERER_PROFILE_BY_TRACK.get(str(renderer_track).strip(), "unknown")


def current_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"
