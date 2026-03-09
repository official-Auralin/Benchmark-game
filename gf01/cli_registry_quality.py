"""Quality and governance command parser registration."""

from __future__ import annotations

import argparse

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
from .meta import (
    ALLOWED_BASELINE_PANEL_LEVELS,
    DEFAULT_MIN_PUBLIC_NOVELTY_RATIO,
    DEFAULT_PRIVATE_EVAL_MIN_COUNT,
    DEFAULT_SPLIT_RATIO_TOLERANCE,
    IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
)


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
        "--unittest-shards",
        type=int,
        default=1,
        help="Split the unittest stage into N balanced shards for faster gate runs",
    )
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
    p_release_governance.add_argument("--out", type=str, default="", help="Optional output JSON path")
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
    p_release_report.add_argument("--runs", type=str, required=True, help="Path to strict run JSONL artifact")
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
    p_release_report.add_argument("--out", type=str, default="", help="Optional output JSON path")
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
    p_release_candidate.add_argument("--freeze-dir", type=str, required=True, help="Directory containing split_manifest_v1.json for the candidate run")
    p_release_candidate.add_argument("--campaign-dir", type=str, required=True, help="Directory containing runs_combined.jsonl for the candidate run")
    p_release_candidate.add_argument("--previous-manifest", type=str, default="", help="Optional previous-cycle manifest used for rotation checks")
    p_release_candidate.add_argument("--require-previous-manifest", action="store_true", help="Fail if --previous-manifest is not provided")
    p_release_candidate.add_argument(
        "--target-ratios",
        type=str,
        default=default_ratio_arg,
        help="Comma-separated split ratios (split=value, normalized internally)",
    )
    p_release_candidate.add_argument("--tolerance", type=float, default=DEFAULT_SPLIT_RATIO_TOLERANCE, help="Absolute per-split ratio tolerance for governance checks")
    p_release_candidate.add_argument("--private-split", type=str, default="private_eval", help="Split id treated as private official-eval split")
    p_release_candidate.add_argument("--min-private-eval-count", type=int, default=DEFAULT_PRIVATE_EVAL_MIN_COUNT, help="Minimum required row count in private split")
    p_release_candidate.add_argument("--min-public-novelty-ratio", type=float, default=DEFAULT_MIN_PUBLIC_NOVELTY_RATIO, help="Minimum novelty ratio in public splits against previous cycle")
    p_release_candidate.add_argument("--allow-non-official-split-names", action="store_true", help="Allow split IDs outside OFFICIAL_SPLITS")
    p_release_candidate.add_argument("--no-strict-manifest", action="store_true", help="Disable strict manifest schema/family checks for governance stage")
    p_release_candidate.add_argument(
        "--baseline-policy-level",
        type=str,
        default="full",
        choices=list(ALLOWED_BASELINE_PANEL_LEVELS),
        help="Required baseline policy level for release report stage",
    )
    p_release_candidate.add_argument("--skip-package", action="store_true", help="Skip reproducibility package assembly stage")
    p_release_candidate.add_argument("--package-out-dir", type=str, default="release_packages/gf01_release_candidate_v1", help="Output directory for release-package stage")
    p_release_candidate.add_argument("--force-package", action="store_true", help="Allow non-empty package output directory overwrite in package stage")
    p_release_candidate.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_release_candidate.set_defaults(func=cmd_release_candidate_check)
