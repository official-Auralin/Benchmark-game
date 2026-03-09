"""Workflow command parser registration."""

from __future__ import annotations

import argparse

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
    ALLOWED_EVAL_TRACKS,
    ALLOWED_MODES,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    GENERATOR_VERSION,
    HARNESS_VERSION,
)


def register_workflow_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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

    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate one baseline agent on external instances",
    )
    p_eval.add_argument("--instances", type=str, required=True, help="Path to instance JSON")
    p_eval.add_argument("--agent", type=str, default="greedy", help="random|greedy|search|tool|oracle")
    p_eval.add_argument(
        "--eval-track",
        type=str,
        default="EVAL-CB",
        choices=list(ALLOWED_EVAL_TRACKS),
    )
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
