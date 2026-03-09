"""Core generation, evaluation, report, validation, and migration commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..baselines import (
    BudgetAwareSearchAgent,
    ExactOracleAgent,
    GreedyLocalAgent,
    RandomInterventionAgent,
    ToolPlannerAgent,
    make_agent,
)
from ..checks import evaluate_suite
from ..generator import generate_instance, generate_suite
from ..io import (
    build_split_manifest,
    load_instance_bundle,
    load_json,
    load_jsonl,
    migrate_run_rows,
    validate_run_rows,
    write_json,
    write_jsonl,
    write_run_records_jsonl,
)
from ..meta import (
    ADAPTATION_POLICY_VERSION,
    ALLOWED_EVAL_TRACKS,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    IDENTIFIABILITY_METRIC_ID,
    IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
    IDENTIFIABILITY_POLICY_VERSION,
    INSTANCE_BUNDLE_SCHEMA_VERSION,
    RENDERER_POLICY_VERSION,
    RUN_RECORD_SCHEMA_VERSION,
    config_hash,
    require_git_commit,
    renderer_profile_for_track,
    stable_hash_json,
)
from ..models import GeneratorConfig
from .shared import (
    adaptation_policy_message,
    build_report_payload,
    renderer_policy_message,
    track_tool_policy_message,
    validate_runs_manifest,
)


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig()
    instance, witness = generate_instance(seed=args.seed, cfg=cfg, split_id="public_dev")
    agents = [
        RandomInterventionAgent(),
        GreedyLocalAgent(),
        BudgetAwareSearchAgent(),
        ToolPlannerAgent(),
        ExactOracleAgent(),
    ]
    out = {
        "instance": instance.to_canonical_dict(),
        "witness": [a.to_tuple() for a in witness],
        "agent_results": {},
    }
    for agent in agents:
        records, agg = evaluate_suite(
            agent=agent,
            instances=[instance],
            eval_track="EVAL-CB" if "Oracle" not in agent.name else "EVAL-OC",
            renderer_track="json",
            seed=args.seed,
        )
        record = records[0]
        out["agent_results"][agent.name] = {
            "valid": record.valid,
            "suff": record.suff,
            "min1": record.min1,
            "eff_t": record.eff_t,
            "eff_a": record.eff_a,
            "ap_f1": record.ap_f1,
            "ts_f1": record.ts_f1,
            "aggregate": agg,
            "certificate": [a.to_tuple() for a in record.certificate],
        }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig()
    seeds = [args.seed + i for i in range(args.count)]
    suite = generate_suite(seeds=seeds, split_id=args.split, cfg=cfg)
    instances_payload = [inst.to_canonical_dict() for inst in suite]
    git_commit = require_git_commit()
    bundle = {
        "schema_version": INSTANCE_BUNDLE_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": git_commit,
        "config_hash": config_hash(cfg),
        "split_id": args.split,
        "seed_start": int(args.seed),
        "count": int(args.count),
        "seeds": seeds,
        "identifiability_policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "identifiability_metric_id": IDENTIFIABILITY_METRIC_ID,
        "identifiability_min_response_ratio": float(IDENTIFIABILITY_MIN_RESPONSE_RATIO),
        "identifiability_min_unique_signatures": int(IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES),
        "instances_hash": stable_hash_json(instances_payload),
        "instances": instances_payload,
    }
    payload = instances_payload if args.legacy_list else bundle
    if args.out:
        if args.legacy_list:
            path = Path(args.out)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            write_json(args.out, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.manifest_out:
        manifest = build_split_manifest(
            suite,
            bundle_meta=(bundle if not args.legacy_list else {}),
        )
        write_json(args.manifest_out, manifest)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    instances, bundle_meta = load_instance_bundle(args.instances)
    agent = make_agent(args.agent)
    eval_track = str(args.eval_track).strip()
    renderer_track = str(args.renderer_track).strip()
    renderer_profile_id = renderer_profile_for_track(renderer_track)
    agent_id = str(args.agent).strip().lower()
    tool_allowlist_id = str(args.tool_allowlist_id).strip()
    tool_log_hash = str(args.tool_log_hash).strip()
    adaptation_condition = str(args.adaptation_condition).strip()
    adaptation_budget_tokens = int(args.adaptation_budget_tokens)
    adaptation_data_scope = str(args.adaptation_data_scope).strip()
    adaptation_protocol_id = str(args.adaptation_protocol_id).strip()

    if eval_track not in ALLOWED_EVAL_TRACKS:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": f"unsupported eval_track {eval_track}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    renderer_msg = renderer_policy_message(
        renderer_track=renderer_track,
        renderer_profile_id=renderer_profile_id,
    )
    if renderer_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "renderer_policy_violation",
                    "message": renderer_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if agent_id in {"tool", "bl-03", "bl-03-toolplanner"} and eval_track == "EVAL-CB":
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": "tool planner agent is not allowed in EVAL-CB",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if agent_id in {"oracle", "bl-04", "bl-04-exactoracle"} and eval_track != "EVAL-OC":
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": "exact oracle agent is restricted to EVAL-OC",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    policy_msg = track_tool_policy_message(
        eval_track=eval_track,
        tool_allowlist_id=tool_allowlist_id,
        tool_log_hash=tool_log_hash,
    )
    if policy_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": policy_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    adaptation_msg = adaptation_policy_message(
        adaptation_condition=adaptation_condition,
        adaptation_budget_tokens=adaptation_budget_tokens,
        adaptation_data_scope=adaptation_data_scope,
        adaptation_protocol_id=adaptation_protocol_id,
    )
    if adaptation_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "adaptation_policy_violation",
                    "message": adaptation_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    run_meta = {
        "family_id": bundle_meta.get("family_id", FAMILY_ID),
        "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
        "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
        "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
        "harness_version": bundle_meta.get("harness_version", HARNESS_VERSION),
        "git_commit": require_git_commit(),
        "config_hash": bundle_meta.get("config_hash", "unknown"),
        "tool_allowlist_id": tool_allowlist_id or "none",
        "tool_log_hash": tool_log_hash,
        "play_protocol": "commit_only",
        "scored_commit_episode": True,
        "renderer_policy_version": RENDERER_POLICY_VERSION,
        "renderer_profile_id": renderer_profile_id,
        "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
    }
    records, aggregate = evaluate_suite(
        agent=agent,
        instances=instances,
        eval_track=eval_track,
        renderer_track=renderer_track,
        seed=args.seed,
    )
    if args.out:
        instance_lookup = {inst.instance_id: inst for inst in instances}
        write_run_records_jsonl(
            args.out,
            records,
            instance_lookup=instance_lookup,
            run_meta=run_meta,
        )
    out = {
        "agent": agent.name,
        "instances": len(instances),
        "eval_track": eval_track,
        "renderer_track": renderer_track,
        "aggregate": aggregate,
        "run_schema_version": RUN_RECORD_SCHEMA_VERSION,
        "source_bundle_schema_version": bundle_meta.get("schema_version", "legacy-instance-list"),
        "tool_allowlist_id": tool_allowlist_id or "none",
        "play_protocol": "commit_only",
        "scored_commit_episode": True,
        "renderer_policy_version": RENDERER_POLICY_VERSION,
        "renderer_profile_id": renderer_profile_id,
        "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
        "out_jsonl": args.out,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    strict_mode = bool(args.strict or args.official)
    err_payload, coverage_payload = validate_runs_manifest(
        rows,
        manifest_path=args.manifest,
        strict_mode=strict_mode,
        official_mode=bool(args.official),
    )
    if err_payload is not None:
        print(json.dumps(err_payload, indent=2, sort_keys=True))
        return 2

    out = build_report_payload(
        rows=rows,
        strict_mode=strict_mode,
        coverage_payload=coverage_payload,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    strict_mode = bool(args.strict or args.official)
    err_payload, coverage_payload = validate_runs_manifest(
        rows,
        manifest_path=args.manifest,
        strict_mode=strict_mode,
        official_mode=bool(args.official),
    )
    if err_payload is not None:
        print(json.dumps(err_payload, indent=2, sort_keys=True))
        return 2
    out: dict[str, object] = {
        "status": "ok",
        "rows": len(rows),
        "strict_mode": strict_mode,
        "schema_versions": sorted({str(r.get("schema_version", "unknown")) for r in rows}),
    }
    if coverage_payload is not None:
        out["manifest_coverage"] = coverage_payload
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_migrate_runs(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    manifest: dict[str, object] | None = load_json(args.manifest) if args.manifest else None
    default_eval_track = str(args.default_eval_track).strip()
    default_renderer_track = str(args.default_renderer_track).strip()
    renderer_profile_id = renderer_profile_for_track(default_renderer_track)
    if default_eval_track not in ALLOWED_EVAL_TRACKS:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": f"unsupported default_eval_track {default_eval_track}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    renderer_msg = renderer_policy_message(
        renderer_track=default_renderer_track,
        renderer_profile_id=renderer_profile_id,
    )
    if renderer_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "renderer_policy_violation",
                    "message": renderer_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    tool_allowlist_id = str(args.tool_allowlist_id).strip()
    tool_log_hash = str(args.tool_log_hash).strip()
    if default_eval_track == "EVAL-CB":
        tool_allowlist_id = "none"
        tool_log_hash = ""
    else:
        if not tool_allowlist_id or tool_allowlist_id.lower() == "none":
            tool_allowlist_id = DEFAULT_TOOL_ALLOWLIST_BY_TRACK[default_eval_track]
        if not tool_log_hash:
            tool_log_hash = stable_hash_json(
                {
                    "migration_out": args.out,
                    "default_eval_track": default_eval_track,
                    "tool_allowlist_id": tool_allowlist_id,
                }
            )[:16]

    policy_msg = track_tool_policy_message(
        eval_track=default_eval_track,
        tool_allowlist_id=tool_allowlist_id,
        tool_log_hash=tool_log_hash,
    )
    if policy_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": policy_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    adaptation_condition = str(args.adaptation_condition).strip()
    adaptation_budget_tokens = int(args.adaptation_budget_tokens)
    adaptation_data_scope = str(args.adaptation_data_scope).strip()
    adaptation_protocol_id = str(args.adaptation_protocol_id).strip()
    adaptation_msg = adaptation_policy_message(
        adaptation_condition=adaptation_condition,
        adaptation_budget_tokens=adaptation_budget_tokens,
        adaptation_data_scope=adaptation_data_scope,
        adaptation_protocol_id=adaptation_protocol_id,
    )
    if adaptation_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "adaptation_policy_violation",
                    "message": adaptation_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    defaults: dict[str, object] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": args.benchmark_version,
        "generator_version": args.generator_version,
        "checker_version": args.checker_version,
        "harness_version": args.harness_version,
        "git_commit": args.git_commit or require_git_commit(),
        "config_hash": args.config_hash,
        "tool_allowlist_id": tool_allowlist_id,
        "tool_log_hash": tool_log_hash,
        "play_protocol": "commit_only",
        "scored_commit_episode": True,
        "renderer_policy_version": RENDERER_POLICY_VERSION,
        "renderer_profile_id": renderer_profile_id,
        "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
        "eval_track": default_eval_track,
        "renderer_track": default_renderer_track,
        "agent_name": args.default_agent_name,
        "split_id": args.default_split_id,
        "mode": args.default_mode,
        "seed": args.default_seed,
    }
    migrated, stats = migrate_run_rows(rows, defaults=defaults, manifest=manifest)
    write_jsonl(args.out, migrated)

    strict_errors = validate_run_rows(migrated, strict=True)
    if strict_errors:
        out_err = {
            "status": "error",
            "error_type": "post_migration_strict_validation",
            "output_path": args.out,
            "error_count": len(strict_errors),
            "errors_preview": strict_errors[:50],
            "migration_stats": stats,
        }
        print(json.dumps(out_err, indent=2, sort_keys=True))
        return 2

    out: dict[str, object] = {
        "status": "ok",
        "output_path": args.out,
        "migration_stats": stats,
    }
    if args.manifest:
        _, coverage_payload = validate_runs_manifest(
            migrated,
            manifest_path=args.manifest,
            strict_mode=False,
            official_mode=False,
        )
        if coverage_payload is not None:
            out["manifest_coverage"] = coverage_payload
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    instances, bundle_meta = load_instance_bundle(args.instances)
    manifest = build_split_manifest(instances, bundle_meta=bundle_meta)
    if args.out:
        write_json(args.out, manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
