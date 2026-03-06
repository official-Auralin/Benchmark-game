"""Parser registration helpers for grouped GF-01 CLI commands."""

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
from .commands.pilot import (
    cmd_freeze_pilot,
    cmd_pilot_analyze,
    cmd_pilot_campaign,
    cmd_release_package,
)
from .commands.playback import cmd_play
from .commands.q033 import (
    cmd_q033_build_manifests,
    cmd_q033_closure_check,
    cmd_q033_sweep,
)
from .commands.quality import (
    cmd_checks,
    cmd_gate,
    cmd_identifiability_check,
    cmd_profile,
    cmd_release_candidate_check,
    cmd_release_governance_check,
    cmd_release_report_check,
    cmd_split_policy_check,
)
from .commands.workflows import (
    cmd_demo,
    cmd_evaluate,
    cmd_generate,
    cmd_manifest,
    cmd_migrate_runs,
    cmd_report,
    cmd_validate,
)
from .meta import (
    ALLOWED_ADAPTATION_CONDITIONS,
    ALLOWED_ADAPTATION_DATA_SCOPES,
    ALLOWED_BASELINE_PANEL_LEVELS,
    ALLOWED_EVAL_TRACKS,
    ALLOWED_MODES,
    ALLOWED_RENDERER_TRACKS,
    ALLOWED_TOOL_ALLOWLISTS_BY_TRACK,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    DEFAULT_MIN_PUBLIC_NOVELTY_RATIO,
    DEFAULT_PRIVATE_EVAL_MIN_COUNT,
    DEFAULT_SPLIT_RATIO_TOLERANCE,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
)


def register_workflow_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_demo = sub.add_parser("demo", help="Run a single-instance baseline demo")
    p_demo.add_argument("--seed", type=int, default=1337)
    p_demo.set_defaults(func=cmd_demo)

    p_gen = sub.add_parser("generate", help="Generate a suite of instances")
    p_gen.add_argument("--seed", type=int, default=2000)
    p_gen.add_argument("--count", type=int, default=4)
    p_gen.add_argument("--split", type=str, default="public_dev")
    p_gen.add_argument("--out", type=str, default="")
    p_gen.add_argument("--manifest-out", type=str, default="")
    p_gen.add_argument(
        "--legacy-list",
        action="store_true",
        help="Emit legacy list-only instance format instead of versioned bundle",
    )
    p_gen.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate", help="Evaluate one baseline agent on external instances")
    p_eval.add_argument("--instances", type=str, required=True, help="Path to instance JSON")
    p_eval.add_argument("--agent", type=str, default="greedy", help="random|greedy|search|tool|oracle")
    p_eval.add_argument("--eval-track", type=str, default="EVAL-CB", choices=list(ALLOWED_EVAL_TRACKS))
    p_eval.add_argument("--renderer-track", type=str, default="json")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--tool-allowlist-id", type=str, default="none")
    p_eval.add_argument("--tool-log-hash", type=str, default="")
    p_eval.add_argument(
        "--adaptation-condition",
        type=str,
        default="no_adaptation",
        choices=list(ALLOWED_ADAPTATION_CONDITIONS),
    )
    p_eval.add_argument("--adaptation-budget-tokens", type=int, default=0)
    p_eval.add_argument(
        "--adaptation-data-scope",
        type=str,
        default="none",
        choices=list(ALLOWED_ADAPTATION_DATA_SCOPES),
    )
    p_eval.add_argument("--adaptation-protocol-id", type=str, default="none")
    p_eval.add_argument("--out", type=str, default="", help="Optional output JSONL path")
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser(
        "report",
        help="Aggregate run JSONL by track/renderer/protocol/split/mode",
    )
    p_report.add_argument("--runs", type=str, required=True, help="Path to run JSONL")
    p_report.add_argument("--manifest", type=str, default="", help="Optional split manifest JSON")
    p_report.add_argument(
        "--strict",
        action="store_true",
        help="Enforce required run/manifest schema fields and strict metadata checks",
    )
    p_report.add_argument(
        "--official",
        action="store_true",
        help="Official reporting mode: implies strict checks and requires --manifest",
    )
    p_report.set_defaults(func=cmd_report)

    p_validate = sub.add_parser(
        "validate",
        help="Validate run JSONL against schema and optional manifest coverage",
    )
    p_validate.add_argument("--runs", type=str, required=True, help="Path to run JSONL")
    p_validate.add_argument("--manifest", type=str, default="", help="Optional split manifest JSON")
    p_validate.add_argument(
        "--strict",
        action="store_true",
        help="Enforce required run/manifest schema fields and strict metadata checks",
    )
    p_validate.add_argument(
        "--official",
        action="store_true",
        help="Official mode: strict checks plus required manifest coverage validation",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_migrate = sub.add_parser(
        "migrate-runs",
        help="Backfill legacy run JSONL rows into gf01.run_record.v1 schema",
    )
    p_migrate.add_argument("--runs", type=str, required=True, help="Path to legacy run JSONL")
    p_migrate.add_argument("--out", type=str, required=True, help="Path to migrated run JSONL")
    p_migrate.add_argument("--manifest", type=str, default="", help="Optional manifest for metadata join")
    p_migrate.add_argument("--benchmark-version", type=str, default=BENCHMARK_VERSION)
    p_migrate.add_argument("--generator-version", type=str, default=GENERATOR_VERSION)
    p_migrate.add_argument("--checker-version", type=str, default=CHECKER_VERSION)
    p_migrate.add_argument("--harness-version", type=str, default=HARNESS_VERSION)
    p_migrate.add_argument("--git-commit", type=str, default="")
    p_migrate.add_argument("--config-hash", type=str, default="legacy-backfill")
    p_migrate.add_argument("--tool-allowlist-id", type=str, default="none")
    p_migrate.add_argument("--tool-log-hash", type=str, default="")
    p_migrate.add_argument(
        "--adaptation-condition",
        type=str,
        default="no_adaptation",
        choices=list(ALLOWED_ADAPTATION_CONDITIONS),
    )
    p_migrate.add_argument("--adaptation-budget-tokens", type=int, default=0)
    p_migrate.add_argument(
        "--adaptation-data-scope",
        type=str,
        default="none",
        choices=list(ALLOWED_ADAPTATION_DATA_SCOPES),
    )
    p_migrate.add_argument("--adaptation-protocol-id", type=str, default="none")
    p_migrate.add_argument(
        "--default-eval-track",
        type=str,
        default="EVAL-CB",
        choices=list(ALLOWED_EVAL_TRACKS),
    )
    p_migrate.add_argument("--default-renderer-track", type=str, default="json")
    p_migrate.add_argument("--default-agent-name", type=str, default="legacy-agent")
    p_migrate.add_argument("--default-split-id", type=str, default="public_dev")
    p_migrate.add_argument(
        "--default-mode",
        type=str,
        default="normal",
        choices=list(ALLOWED_MODES),
    )
    p_migrate.add_argument("--default-seed", type=int, default=0)
    p_migrate.set_defaults(func=cmd_migrate_runs)

    p_manifest = sub.add_parser("manifest", help="Build split manifest from an instance file")
    p_manifest.add_argument("--instances", type=str, required=True, help="Path to instance JSON")
    p_manifest.add_argument("--out", type=str, default="", help="Optional output manifest path")
    p_manifest.set_defaults(func=cmd_manifest)


def register_quality_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    default_ratio_arg: str,
) -> None:
    p_chk = sub.add_parser("checks", help="Run priority H3 checks AB-001/003/005/007/009")
    p_chk.add_argument("--seed", type=int, default=3000)
    p_chk.set_defaults(func=cmd_checks)

    p_prof = sub.add_parser("profile", help="Run Python-first profiler and gate checks")
    p_prof.add_argument("--seed", type=int, default=4000)
    p_prof.add_argument("--public-count", type=int, default=3)
    p_prof.add_argument("--private-count", type=int, default=3)
    p_prof.add_argument(
        "--cprofile-out",
        type=str,
        default="",
        help="Optional cProfile output path (e.g., prof.stats)",
    )
    p_prof.set_defaults(func=cmd_profile)

    p_gate = sub.add_parser(
        "gate",
        help="Run one-shot CI-style regression gate (compile, tests, checks, profile, fixture validation)",
    )
    p_gate.add_argument(
        "--fixture-root",
        type=str,
        default="tests/fixtures/official_example",
        help="Directory containing runs_v1_valid.jsonl and split_manifest_v1.json",
    )
    p_gate.add_argument("--seed-checks", type=int, default=3000)
    p_gate.add_argument("--seed-profile", type=int, default=4000)
    p_gate.add_argument("--public-count", type=int, default=3)
    p_gate.add_argument("--private-count", type=int, default=3)
    p_gate.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the gate immediately on first failed step",
    )
    p_gate.set_defaults(func=cmd_gate)

    p_split_policy = sub.add_parser(
        "split-policy-check",
        help="Validate a split manifest against publication split-ratio policy",
    )
    p_split_policy.add_argument("--manifest", type=str, required=True, help="Path to split manifest JSON")
    p_split_policy.add_argument(
        "--target-ratios",
        type=str,
        default=default_ratio_arg,
        help="Comma-separated split ratios (split=value, normalized internally)",
    )
    p_split_policy.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_SPLIT_RATIO_TOLERANCE,
        help="Absolute per-split ratio tolerance",
    )
    p_split_policy.add_argument(
        "--private-split",
        type=str,
        default="private_eval",
        help="Split id treated as private official-eval split",
    )
    p_split_policy.add_argument(
        "--min-private-eval-count",
        type=int,
        default=DEFAULT_PRIVATE_EVAL_MIN_COUNT,
        help="Minimum required row count in private split",
    )
    p_split_policy.add_argument(
        "--require-official-split-names",
        action="store_true",
        help="Require split IDs to be in OFFICIAL_SPLITS",
    )
    p_split_policy.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Apply strict manifest schema/family checks before ratio checks",
    )
    p_split_policy.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_split_policy.set_defaults(func=cmd_split_policy_check)

    p_release_governance = sub.add_parser(
        "release-governance-check",
        help=(
            "Validate release split policy plus seed/instance-rotation "
            "contamination safeguards (machine-checkable)."
        ),
    )
    p_release_governance.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to current split manifest JSON",
    )
    p_release_governance.add_argument(
        "--previous-manifest",
        type=str,
        default="",
        help="Optional path to previous-cycle split manifest JSON",
    )
    p_release_governance.add_argument(
        "--require-previous-manifest",
        action="store_true",
        help="Fail if --previous-manifest is not provided",
    )
    p_release_governance.add_argument(
        "--target-ratios",
        type=str,
        default=default_ratio_arg,
        help="Comma-separated split ratios (split=value, normalized internally)",
    )
    p_release_governance.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_SPLIT_RATIO_TOLERANCE,
        help="Absolute per-split ratio tolerance",
    )
    p_release_governance.add_argument(
        "--private-split",
        type=str,
        default="private_eval",
        help="Split id treated as private official-eval split",
    )
    p_release_governance.add_argument(
        "--min-private-eval-count",
        type=int,
        default=DEFAULT_PRIVATE_EVAL_MIN_COUNT,
        help="Minimum required row count in private split",
    )
    p_release_governance.add_argument(
        "--min-public-novelty-ratio",
        type=float,
        default=DEFAULT_MIN_PUBLIC_NOVELTY_RATIO,
        help=(
            "Minimum ratio of current public instances not present in previous "
            "public set; ignored when no previous manifest is provided."
        ),
    )
    p_release_governance.add_argument(
        "--allow-non-official-split-names",
        action="store_true",
        help="Allow split IDs outside OFFICIAL_SPLITS (default is strict names)",
    )
    p_release_governance.add_argument(
        "--no-strict-manifest",
        action="store_true",
        help="Disable strict manifest schema/family checks",
    )
    p_release_governance.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_release_governance.set_defaults(
        func=cmd_release_governance_check,
        strict_manifest=True,
        require_official_split_names=True,
    )

    p_release_report = sub.add_parser(
        "release-report-check",
        help=(
            "Validate release baseline-panel and per-track/per-slice reporting "
            "coverage from strict run artifacts."
        ),
    )
    p_release_report.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Path to strict run JSONL artifact",
    )
    p_release_report.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to split manifest JSON used for expected slice coverage",
    )
    p_release_report.add_argument(
        "--baseline-policy-level",
        type=str,
        default="full",
        choices=list(ALLOWED_BASELINE_PANEL_LEVELS),
        help=(
            "Required baseline policy level for release report coverage checks "
            "(full or core)."
        ),
    )
    p_release_report.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_release_report.set_defaults(func=cmd_release_report_check)

    p_ident = sub.add_parser(
        "identifiability-check",
        help="Validate instance bundles against partial-observability identifiability thresholds",
    )
    p_ident.add_argument("--instances", type=str, required=True, help="Path to instance bundle/list JSON")
    p_ident.add_argument(
        "--min-response-ratio",
        type=float,
        default=IDENTIFIABILITY_MIN_RESPONSE_RATIO,
        help="Minimum required single-atom observable response ratio",
    )
    p_ident.add_argument(
        "--min-unique-signatures",
        type=int,
        default=IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
        help="Minimum required number of unique observable signatures",
    )
    p_ident.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_ident.set_defaults(func=cmd_identifiability_check)

    p_release_candidate = sub.add_parser(
        "release-candidate-check",
        help=(
            "Run release governance/report/package checks end-to-end from a "
            "frozen pack and campaign artifact directory."
        ),
    )
    p_release_candidate.add_argument(
        "--freeze-dir",
        type=str,
        required=True,
        help="Directory containing split_manifest_v1.json for the candidate run",
    )
    p_release_candidate.add_argument(
        "--campaign-dir",
        type=str,
        required=True,
        help="Directory containing runs_combined.jsonl for the candidate run",
    )
    p_release_candidate.add_argument(
        "--previous-manifest",
        type=str,
        default="",
        help="Optional previous-cycle manifest used for rotation checks",
    )
    p_release_candidate.add_argument(
        "--require-previous-manifest",
        action="store_true",
        help="Fail if --previous-manifest is not provided",
    )
    p_release_candidate.add_argument(
        "--target-ratios",
        type=str,
        default=default_ratio_arg,
        help="Comma-separated split ratios (split=value, normalized internally)",
    )
    p_release_candidate.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_SPLIT_RATIO_TOLERANCE,
        help="Absolute per-split ratio tolerance for governance checks",
    )
    p_release_candidate.add_argument(
        "--private-split",
        type=str,
        default="private_eval",
        help="Split id treated as private official-eval split",
    )
    p_release_candidate.add_argument(
        "--min-private-eval-count",
        type=int,
        default=DEFAULT_PRIVATE_EVAL_MIN_COUNT,
        help="Minimum required row count in private split",
    )
    p_release_candidate.add_argument(
        "--min-public-novelty-ratio",
        type=float,
        default=DEFAULT_MIN_PUBLIC_NOVELTY_RATIO,
        help="Minimum novelty ratio in public splits against previous cycle",
    )
    p_release_candidate.add_argument(
        "--allow-non-official-split-names",
        action="store_true",
        help="Allow split IDs outside OFFICIAL_SPLITS",
    )
    p_release_candidate.add_argument(
        "--no-strict-manifest",
        action="store_true",
        help="Disable strict manifest schema/family checks for governance stage",
    )
    p_release_candidate.add_argument(
        "--baseline-policy-level",
        type=str,
        default="full",
        choices=list(ALLOWED_BASELINE_PANEL_LEVELS),
        help="Required baseline policy level for release report stage",
    )
    p_release_candidate.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip reproducibility package assembly stage",
    )
    p_release_candidate.add_argument(
        "--package-out-dir",
        type=str,
        default="release_packages/gf01_release_candidate_v1",
        help="Output directory for release-package stage",
    )
    p_release_candidate.add_argument(
        "--force-package",
        action="store_true",
        help="Allow non-empty package output directory overwrite in package stage",
    )
    p_release_candidate.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_release_candidate.set_defaults(func=cmd_release_candidate_check)


def register_q033_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_q033_manifest = sub.add_parser(
        "q033-build-manifests",
        help="Build deterministic balanced quartile seed manifests for Q-033 sweeps",
    )
    p_q033_manifest.add_argument("--seed-start", type=int, default=8000)
    p_q033_manifest.add_argument("--candidate-count", type=int, default=4000)
    p_q033_manifest.add_argument("--replicates", type=int, default=2)
    p_q033_manifest.add_argument("--per-quartile", type=int, default=120)
    p_q033_manifest.add_argument("--split", type=str, default="q033_internal")
    p_q033_manifest.add_argument(
        "--out-dir",
        type=str,
        default="q033_manifests/q033_protocol_v1",
    )
    p_q033_manifest.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_q033_manifest.set_defaults(func=cmd_q033_build_manifests)

    p_q033_sweep = sub.add_parser(
        "q033-sweep",
        help="Run one Q-033 profiling sweep replicate from a seed manifest",
    )
    p_q033_sweep.add_argument("--manifest", type=str, required=True)
    p_q033_sweep.add_argument(
        "--baseline-panel",
        type=str,
        default="random,greedy,search,tool,oracle",
    )
    p_q033_sweep.add_argument("--seed", type=int, default=1300)
    p_q033_sweep.add_argument("--max-generate-ms-mean", type=float, default=1200.0)
    p_q033_sweep.add_argument("--max-minset-ms-mean", type=float, default=2500.0)
    p_q033_sweep.add_argument("--max-eval-ms-mean", type=float, default=1500.0)
    p_q033_sweep.add_argument("--max-checks-total-ms", type=float, default=30000.0)
    p_q033_sweep.add_argument("--max-truncation-rate", type=float, default=0.25)
    p_q033_sweep.add_argument("--min-oracle-minus-greedy-gap", type=float, default=0.10)
    p_q033_sweep.add_argument("--max-quartile-truncation-rate", type=float, default=0.30)
    p_q033_sweep.add_argument("--min-quartile-gap", type=float, default=0.05)
    p_q033_sweep.add_argument("--max-quartile-runtime-gate-failures", type=int, default=1)
    p_q033_sweep.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_q033_sweep.set_defaults(func=cmd_q033_sweep)

    p_q033_close = sub.add_parser(
        "q033-closure-check",
        help="Check Q-033 closure rule from two or more sweep outputs",
    )
    p_q033_close.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Path to one q033-sweep output JSON (repeat flag for multiple replicates)",
    )
    p_q033_close.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help="Disable disjoint-seed requirement across replicates",
    )
    p_q033_close.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_q033_close.set_defaults(func=cmd_q033_closure_check)


def register_pilot_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_freeze = sub.add_parser(
        "freeze-pilot",
        help="Freeze a provisional internal pilot pack (bundle + manifest + freeze metadata)",
    )
    p_freeze.add_argument("--freeze-id", type=str, default="gf01-pilot-freeze-v1")
    p_freeze.add_argument("--split", type=str, default="pilot_internal_v1")
    p_freeze.add_argument("--seed-start", type=int, default=7000)
    p_freeze.add_argument("--count", type=int, default=24)
    p_freeze.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated explicit seed list; overrides --seed-start/--count",
    )
    p_freeze.add_argument(
        "--mode",
        type=str,
        default="",
        help=(
            "Optional mode override for all frozen instances (normal/hard). "
            "Leave empty for mixed generation."
        ),
    )
    p_freeze.add_argument("--out-dir", type=str, default="pilot_freeze/gf01_pilot_freeze_v1")
    p_freeze.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_freeze.set_defaults(func=cmd_freeze_pilot)

    p_campaign = sub.add_parser(
        "pilot-campaign",
        help="Run a pilot campaign on a frozen pack with official validation/report outputs",
    )
    p_campaign.add_argument(
        "--freeze-dir",
        type=str,
        required=True,
        help="Directory containing instance_bundle_v1.json and split_manifest_v1.json",
    )
    p_campaign.add_argument(
        "--out-dir",
        type=str,
        default="pilot_runs/gf01_pilot_campaign_v1",
        help="Output directory for combined runs, validation, and report artifacts",
    )
    p_campaign.add_argument(
        "--baseline-panel",
        type=str,
        default="random,greedy,search,tool,oracle",
        help="Comma-separated baseline ids (e.g., random,greedy,search,tool,oracle)",
    )
    p_campaign.add_argument(
        "--baseline-policy-level",
        type=str,
        default="full",
        choices=list(ALLOWED_BASELINE_PANEL_LEVELS),
        help=(
            "Baseline-policy enforcement level: "
            "'full' requires random,greedy,search,tool,oracle; "
            "'core' requires random,greedy,oracle."
        ),
    )
    p_campaign.add_argument(
        "--renderer-track",
        type=str,
        default="json",
        choices=list(ALLOWED_RENDERER_TRACKS),
    )
    p_campaign.add_argument("--seed", type=int, default=1100)
    p_campaign.add_argument(
        "--tool-allowlist-id",
        type=str,
        default=DEFAULT_TOOL_ALLOWLIST_BY_TRACK["EVAL-TA"],
        choices=list(ALLOWED_TOOL_ALLOWLISTS_BY_TRACK["EVAL-TA"]),
        help="Tool allowlist id used for EVAL-TA baseline rows",
    )
    p_campaign.add_argument("--tool-log-hash", type=str, default="")
    p_campaign.add_argument(
        "--external-runs",
        action="append",
        default=[],
        help="Optional path to external run JSONL (repeatable)",
    )
    p_campaign.add_argument(
        "--external-episodes",
        action="append",
        default=[],
        help=(
            "Optional play-output artifact path (JSON or JSONL). "
            "Each payload is converted to strict run rows and merged."
        ),
    )
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
    p_campaign.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_campaign.set_defaults(func=cmd_pilot_campaign)

    p_release_pkg = sub.add_parser(
        "release-package",
        help=(
            "Build a reproducibility package from a frozen pack and campaign "
            "artifacts with strict validation."
        ),
    )
    p_release_pkg.add_argument(
        "--freeze-dir",
        type=str,
        required=True,
        help="Directory containing frozen pilot artifacts (bundle + manifest)",
    )
    p_release_pkg.add_argument(
        "--campaign-dir",
        type=str,
        required=True,
        help="Directory containing campaign artifacts (runs + validation + report)",
    )
    p_release_pkg.add_argument(
        "--out-dir",
        type=str,
        default="release_packages/gf01_release_package_v1",
        help="Output directory for packaged artifacts and manifest",
    )
    p_release_pkg.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_release_pkg.set_defaults(func=cmd_release_package)

    p_analysis = sub.add_parser(
        "pilot-analyze",
        help="Analyze a pilot campaign and evaluate DEC-014d calibration triggers",
    )
    p_analysis.add_argument(
        "--campaign-dir",
        type=str,
        required=True,
        help="Directory containing campaign artifacts (runs_combined.jsonl at minimum)",
    )
    p_analysis.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output path for analysis JSON (default: <campaign-dir>/pilot_analysis.json)",
    )
    p_analysis.add_argument("--eval-track", type=str, default="EVAL-CB")
    p_analysis.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=list(ALLOWED_MODES),
    )
    p_analysis.add_argument(
        "--greedy-agent-name",
        type=str,
        default="BL-01-GreedyLocal",
    )
    p_analysis.add_argument(
        "--public-splits",
        type=str,
        default="public_dev,public_val",
        help="Comma-separated split IDs treated as public (excluded from held-out shortcut check)",
    )
    p_analysis.add_argument("--sample-target", type=int, default=240)
    p_analysis.add_argument("--discrimination-delta-threshold", type=float, default=0.12)
    p_analysis.add_argument("--discrimination-q4-floor", type=float, default=0.10)
    p_analysis.add_argument("--shortcut-goal-threshold", type=float, default=0.40)
    p_analysis.add_argument("--shortcut-certified-floor", type=float, default=0.05)
    p_analysis.set_defaults(func=cmd_pilot_analyze)


def register_playback_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_play = sub.add_parser(
        "play",
        help="Run one playable GF-01 episode (human, scripted, or baseline agent)",
    )
    p_play.add_argument("--seed", type=int, default=1337)
    p_play.add_argument("--split", type=str, default="public_dev")
    p_play.add_argument("--instances", type=str, default="", help="Optional instance bundle/list JSON")
    p_play.add_argument(
        "--instance-index",
        type=int,
        default=0,
        help="Instance index when --instances contains multiple entries",
    )
    p_play.add_argument("--agent", type=str, default="", help="Optional baseline policy id")
    p_play.add_argument("--script", type=str, default="", help="Optional action script JSON")
    p_play.add_argument(
        "--renderer-track",
        type=str,
        choices=list(ALLOWED_RENDERER_TRACKS),
        default="visual",
    )
    p_play.add_argument(
        "--visual-backend",
        type=str,
        default="text",
        choices=["text", "pygame"],
        help=(
            "Human-play visual backend. 'text' uses terminal snapshots; "
            "'pygame' opens a map-first graphical window when available."
        ),
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


def register_p0_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_p0 = sub.add_parser(
        "p0-feedback-check",
        help="Evaluate internal-alpha (P0) feedback CSV against clarity/blocker thresholds",
    )
    p_p0.add_argument(
        "--feedback",
        type=str,
        required=True,
        help="CSV with required columns: tester_id, objective_clarity, control_clarity, action_effect_clarity, must_fix_blockers",
    )
    p_p0.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum per-tester clarity score counted as acceptable",
    )
    p_p0.add_argument(
        "--min-ratio",
        type=float,
        default=0.80,
        help="Minimum acceptable ratio for each clarity metric",
    )
    p_p0.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_p0.set_defaults(func=cmd_p0_feedback_check)

    p_p0_session = sub.add_parser(
        "p0-session-check",
        help="Verify P0 play-session artifacts exist and match feedback seed/backend declarations",
    )
    p_p0_session.add_argument(
        "--feedback",
        type=str,
        required=True,
        help="Feedback CSV (must include tester_id, backend_used, seed_list_run)",
    )
    p_p0_session.add_argument(
        "--runs-dir",
        type=str,
        default="p0_runs",
        help="Directory containing per-session play artifacts named <tester_id>_<seed>.json",
    )
    p_p0_session.add_argument(
        "--required-renderer-track",
        type=str,
        default="visual",
        choices=list(ALLOWED_RENDERER_TRACKS),
        help="Expected renderer_track in each play artifact run_contract",
    )
    p_p0_session.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_p0_session.set_defaults(func=cmd_p0_session_check)

    p_p0_gate = sub.add_parser(
        "p0-gate",
        help=(
            "Run P0 session-coverage and feedback-threshold checks in sequence "
            "and emit a single pass/fail summary"
        ),
    )
    p_p0_gate.add_argument(
        "--feedback",
        type=str,
        required=True,
        help="Feedback CSV used for both session and feedback checks",
    )
    p_p0_gate.add_argument(
        "--runs-dir",
        type=str,
        default="p0_runs",
        help="Directory containing per-session play artifacts named <tester_id>_<seed>.json",
    )
    p_p0_gate.add_argument(
        "--required-renderer-track",
        type=str,
        default="visual",
        choices=list(ALLOWED_RENDERER_TRACKS),
        help="Expected renderer_track in each play artifact run_contract",
    )
    p_p0_gate.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum per-tester clarity score counted as acceptable",
    )
    p_p0_gate.add_argument(
        "--min-ratio",
        type=float,
        default=0.80,
        help="Minimum acceptable ratio for each clarity metric",
    )
    p_p0_gate.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_p0_gate.set_defaults(func=cmd_p0_gate)

    p_p0_template = sub.add_parser(
        "p0-feedback-template",
        help="Write a deterministic CSV template for P0 internal-alpha feedback collection",
    )
    p_p0_template.add_argument(
        "--out",
        type=str,
        default="p0_feedback_template.csv",
        help="Path to write feedback CSV template",
    )
    p_p0_template.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    p_p0_template.set_defaults(func=cmd_p0_feedback_template)

    p_p0_seed = sub.add_parser(
        "p0-seed-pack",
        help="Freeze a deterministic P0 internal-alpha seed pack using canonical defaults",
    )
    p_p0_seed.add_argument("--freeze-id", type=str, default="gf01-p0-alpha-v1")
    p_p0_seed.add_argument("--split", type=str, default="pilot_internal_p0_v1")
    p_p0_seed.add_argument("--seed-start", type=int, default=7000)
    p_p0_seed.add_argument("--count", type=int, default=8)
    p_p0_seed.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional explicit comma-separated seed list; overrides --seed-start/--count",
    )
    p_p0_seed.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=list(ALLOWED_MODES),
        help="P0 default is normal mode; override only for targeted checks",
    )
    p_p0_seed.add_argument(
        "--out-dir",
        type=str,
        default="research_pack/pilot_freeze/gf01_p0_alpha_v1",
        help="Output directory for the frozen P0 pack",
    )
    p_p0_seed.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_p0_seed.set_defaults(func=cmd_p0_seed_pack)

    p_p0_init = sub.add_parser(
        "p0-init",
        help="One-shot P0 setup: feedback template plus deterministic seed pack",
    )
    p_p0_init.add_argument(
        "--template-out",
        type=str,
        default="p0_feedback.csv",
        help="Path to write P0 feedback CSV template",
    )
    p_p0_init.add_argument("--freeze-id", type=str, default="gf01-p0-alpha-v1")
    p_p0_init.add_argument("--split", type=str, default="pilot_internal_p0_v1")
    p_p0_init.add_argument("--seed-start", type=int, default=7000)
    p_p0_init.add_argument("--count", type=int, default=8)
    p_p0_init.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional explicit comma-separated seed list; overrides --seed-start/--count",
    )
    p_p0_init.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=list(ALLOWED_MODES),
        help="P0 default is normal mode; override only for targeted checks",
    )
    p_p0_init.add_argument(
        "--out-dir",
        type=str,
        default="research_pack/pilot_freeze/gf01_p0_alpha_v1",
        help="Output directory for the frozen P0 pack",
    )
    p_p0_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite template output and non-empty P0 seed-pack directory",
    )
    p_p0_init.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    p_p0_init.set_defaults(func=cmd_p0_init)
