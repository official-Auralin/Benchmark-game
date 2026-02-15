"""
Command-line entry points for generating, evaluating, and reporting GF-01 runs.

This module wires user-facing commands to the underlying harness components.
It is the main operational interface for local demos, benchmark checks,
profiling, external instance evaluation, and summary reporting.
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

import argparse
import cProfile
import json
import statistics
from datetime import date
from pathlib import Path

from .baselines import (
    BudgetAwareSearchAgent,
    ExactOracleAgent,
    GreedyLocalAgent,
    RandomInterventionAgent,
    ToolPlannerAgent,
    make_agent,
)
from .checks import evaluate_suite, run_priority_h3_checks
from .generator import generate_instance, generate_suite
from .gate import run_regression_gate
from .io import (
    build_split_manifest,
    load_instance_bundle,
    load_json,
    load_jsonl,
    migrate_run_rows,
    run_record_to_dict,
    validate_manifest,
    validate_run_rows,
    write_json,
    write_jsonl,
    write_run_records_jsonl,
)
from .meta import (
    ALLOWED_EVAL_TRACKS,
    ALLOWED_MODES,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    INSTANCE_BUNDLE_SCHEMA_VERSION,
    PILOT_FREEZE_SCHEMA_VERSION,
    RUN_RECORD_SCHEMA_VERSION,
    config_hash,
    current_git_commit,
    stable_hash_json,
)
from .models import GeneratorConfig
from .play import baseline_policy, human_policy, parse_action_script, run_episode, scripted_policy
from .profiling import profile_pipeline


def _compute_manifest_coverage(
    rows: list[dict[str, object]], manifest: dict[str, object]
) -> tuple[dict[str, object], list[str], list[str]]:
    manifest_entries = manifest.get("instances", [])
    expected_ids = {str(e.get("instance_id")) for e in manifest_entries}
    observed_ids = {str(r.get("instance_id")) for r in rows}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)

    expected_group_counts: dict[tuple[str, str], int] = {}
    for entry in manifest_entries:
        key = (str(entry.get("split_id", "unknown")), str(entry.get("mode", "unknown")))
        expected_group_counts[key] = expected_group_counts.get(key, 0) + 1

    observed_group_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("split_id", "unknown")), str(row.get("mode", "unknown")))
        observed_group_counts[key] = observed_group_counts.get(key, 0) + 1

    coverage = []
    for key in sorted(set(expected_group_counts) | set(observed_group_counts)):
        split_id, mode = key
        expected = expected_group_counts.get(key, 0)
        observed = observed_group_counts.get(key, 0)
        coverage.append(
            {
                "split_id": split_id,
                "mode": mode,
                "expected": expected,
                "observed": observed,
                "coverage_rate": observed / max(1, expected),
            }
        )

    payload = {
        "manifest_schema_version": manifest.get("schema_version", "unknown"),
        "manifest_instance_count": len(expected_ids),
        "observed_instance_count": len(observed_ids),
        "missing_instance_count": len(missing_ids),
        "unexpected_instance_count": len(unexpected_ids),
        "missing_instance_ids_preview": missing_ids[:10],
        "unexpected_instance_ids_preview": unexpected_ids[:10],
        "group_coverage": coverage,
    }
    return payload, missing_ids, unexpected_ids


def _validate_runs_manifest(
    rows: list[dict[str, object]],
    *,
    manifest_path: str = "",
    strict_mode: bool = False,
    official_mode: bool = False,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    row_errors = validate_run_rows(rows, strict=strict_mode)
    if row_errors:
        return (
            {
                "status": "error",
                "error_type": "run_schema_validation",
                "strict_mode": strict_mode,
                "error_count": len(row_errors),
                "errors_preview": row_errors[:50],
            },
            None,
        )

    if official_mode and not manifest_path:
        return (
            {
                "status": "error",
                "error_type": "official_manifest_required",
                "message": "official mode requires --manifest for coverage validation",
            },
            None,
        )

    coverage: dict[str, object] | None = None
    if manifest_path:
        manifest = load_json(manifest_path)
        manifest_errors = validate_manifest(manifest, strict=strict_mode)
        if manifest_errors:
            return (
                {
                    "status": "error",
                    "error_type": "manifest_schema_validation",
                    "strict_mode": strict_mode,
                    "error_count": len(manifest_errors),
                    "errors_preview": manifest_errors[:50],
                },
                None,
            )
        coverage, missing_ids, unexpected_ids = _compute_manifest_coverage(rows, manifest)
        if strict_mode and (missing_ids or unexpected_ids):
            return (
                {
                    "status": "error",
                    "error_type": "manifest_coverage_mismatch",
                    "strict_mode": strict_mode,
                    "missing_instance_count": len(missing_ids),
                    "unexpected_instance_count": len(unexpected_ids),
                    "missing_instance_ids_preview": missing_ids[:10],
                    "unexpected_instance_ids_preview": unexpected_ids[:10],
                },
                coverage,
            )
    return None, coverage


def _cmd_demo(args: argparse.Namespace) -> int:
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


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig()
    seeds = [args.seed + i for i in range(args.count)]
    suite = generate_suite(seeds=seeds, split_id=args.split, cfg=cfg)
    instances_payload = [inst.to_canonical_dict() for inst in suite]
    bundle = {
        "schema_version": INSTANCE_BUNDLE_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": current_git_commit(),
        "config_hash": config_hash(cfg),
        "split_id": args.split,
        "seed_start": int(args.seed),
        "count": int(args.count),
        "seeds": seeds,
        "generated_on": date.today().isoformat(),
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
            source_path=(args.out or "stdout"),
        )
        write_json(args.manifest_out, manifest)
    return 0


def _cmd_checks(args: argparse.Namespace) -> int:
    results = run_priority_h3_checks(seed_base=args.seed)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig()

    def _run() -> dict[str, object]:
        return profile_pipeline(
            seed_base=args.seed,
            public_count=args.public_count,
            private_count=args.private_count,
            cfg=cfg,
        )

    if args.cprofile_out:
        profiler = cProfile.Profile()
        profiler.enable()
        report = _run()
        profiler.disable()
        profiler.dump_stats(args.cprofile_out)
    else:
        report = _run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    instances, bundle_meta = load_instance_bundle(args.instances)
    agent = make_agent(args.agent)
    eval_track = args.eval_track
    renderer_track = args.renderer_track
    run_meta = {
        "family_id": bundle_meta.get("family_id", FAMILY_ID),
        "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
        "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
        "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
        "harness_version": bundle_meta.get("harness_version", HARNESS_VERSION),
        "git_commit": current_git_commit(),
        "config_hash": bundle_meta.get("config_hash", "unknown"),
        "tool_allowlist_id": args.tool_allowlist_id,
        "tool_log_hash": args.tool_log_hash,
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
        "tool_allowlist_id": args.tool_allowlist_id,
        "out_jsonl": args.out,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    strict_mode = bool(args.strict or args.official)
    err_payload, coverage_payload = _validate_runs_manifest(
        rows,
        manifest_path=args.manifest,
        strict_mode=strict_mode,
        official_mode=bool(args.official),
    )
    if err_payload is not None:
        print(json.dumps(err_payload, indent=2, sort_keys=True))
        return 2

    out = _build_report_payload(
        rows=rows,
        strict_mode=strict_mode,
        coverage_payload=coverage_payload,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _build_report_payload(
    *,
    rows: list[dict[str, object]],
    strict_mode: bool,
    coverage_payload: dict[str, object] | None,
) -> dict[str, object]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("eval_track", "unknown")),
            str(row.get("renderer_track", "unknown")),
            str(row.get("split_id", "unknown")),
            str(row.get("mode", "unknown")),
        )
        groups.setdefault(key, []).append(row)

    report_rows = []
    for key, entries in sorted(groups.items()):
        eval_track, renderer_track, split_id, mode = key
        n = len(entries)
        cert_rate = sum(1 for e in entries if bool(e.get("valid", False))) / max(1, n)
        goal_rate = sum(1 for e in entries if bool(e.get("goal", False))) / max(1, n)
        ap_f1_mean = statistics.fmean(float(e.get("ap_f1", 0.0)) for e in entries)
        ts_f1_mean = statistics.fmean(float(e.get("ts_f1", 0.0)) for e in entries)
        report_rows.append(
            {
                "eval_track": eval_track,
                "renderer_track": renderer_track,
                "split_id": split_id,
                "mode": mode,
                "count": n,
                "certified_rate": cert_rate,
                "goal_rate": goal_rate,
                "ap_f1_mean": ap_f1_mean,
                "ts_f1_mean": ts_f1_mean,
            }
        )
    schema_versions = sorted({str(r.get("schema_version", "unknown")) for r in rows})

    out: dict[str, object] = {
        "groups": report_rows,
        "schema_versions": schema_versions,
        "strict_mode": strict_mode,
        "total_rows": len(rows),
    }
    if coverage_payload is not None:
        out["manifest_coverage"] = coverage_payload
    return out


def _cmd_validate(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    strict_mode = bool(args.strict or args.official)
    err_payload, coverage_payload = _validate_runs_manifest(
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


def _cmd_migrate_runs(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    manifest: dict[str, object] | None = load_json(args.manifest) if args.manifest else None
    defaults: dict[str, object] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": args.benchmark_version,
        "generator_version": args.generator_version,
        "checker_version": args.checker_version,
        "harness_version": args.harness_version,
        "git_commit": args.git_commit or current_git_commit(),
        "config_hash": args.config_hash,
        "tool_allowlist_id": args.tool_allowlist_id,
        "tool_log_hash": args.tool_log_hash,
        "eval_track": args.default_eval_track,
        "renderer_track": args.default_renderer_track,
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
        _, coverage_payload = _validate_runs_manifest(
            migrated,
            manifest_path=args.manifest,
            strict_mode=False,
            official_mode=False,
        )
        if coverage_payload is not None:
            out["manifest_coverage"] = coverage_payload
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    summary = run_regression_gate(
        root=Path(".").resolve(),
        fixture_root=Path(args.fixture_root),
        seed_checks=int(args.seed_checks),
        seed_profile=int(args.seed_profile),
        public_count=int(args.public_count),
        private_count=int(args.private_count),
        fail_fast=bool(args.fail_fast),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary.get("passed", False)) else 2


def _cmd_manifest(args: argparse.Namespace) -> int:
    instances, bundle_meta = load_instance_bundle(args.instances)
    manifest = build_split_manifest(instances, bundle_meta=bundle_meta, source_path=args.instances)
    if args.out:
        write_json(args.out, manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parse_seed_list(seed_text: str) -> list[int]:
    seeds: list[int] = []
    for part in seed_text.split(","):
        token = part.strip()
        if not token:
            continue
        seeds.append(int(token))
    deduped: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed not in seen:
            deduped.append(seed)
            seen.add(seed)
    return deduped


def _cmd_freeze_pilot(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig()
    if args.seeds.strip():
        seeds = _parse_seed_list(args.seeds)
    else:
        seeds = [args.seed_start + i for i in range(args.count)]
    if not seeds:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "empty_seed_set",
                    "message": "pilot freeze requires at least one seed",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "output_dir_not_empty",
                    "message": "output directory exists and is not empty; use --force to overwrite",
                    "out_dir": str(out_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "instance_bundle_v1.json"
    manifest_path = out_dir / "split_manifest_v1.json"
    freeze_path = out_dir / "pilot_freeze_v1.json"

    suite = generate_suite(seeds=seeds, split_id=args.split, cfg=cfg)
    instances_payload = [inst.to_canonical_dict() for inst in suite]
    bundle = {
        "schema_version": INSTANCE_BUNDLE_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": current_git_commit(),
        "config_hash": config_hash(cfg),
        "split_id": args.split,
        "seed_start": int(min(seeds)),
        "count": len(seeds),
        "seeds": seeds,
        "generated_on": date.today().isoformat(),
        "instances_hash": stable_hash_json(instances_payload),
        "instances": instances_payload,
    }
    write_json(str(bundle_path), bundle)

    manifest = build_split_manifest(
        suite,
        bundle_meta=bundle,
        source_path=str(bundle_path),
    )
    manifest_errors = validate_manifest(manifest, strict=True)
    if manifest_errors:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "manifest_validation_failed",
                    "errors": manifest_errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    write_json(str(manifest_path), manifest)

    freeze_meta = {
        "schema_version": PILOT_FREEZE_SCHEMA_VERSION,
        "freeze_id": args.freeze_id,
        "provisional": True,
        "purpose": "internal_pilot",
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": current_git_commit(),
        "config_hash": config_hash(cfg),
        "split_id": args.split,
        "seed_count": len(seeds),
        "seeds": seeds,
        "created_on": date.today().isoformat(),
        "instance_bundle_path": str(bundle_path),
        "split_manifest_path": str(manifest_path),
        "instance_bundle_hash": stable_hash_json(bundle),
        "split_manifest_hash": stable_hash_json(manifest),
        "notes": (
            "Provisional pilot freeze for internal testing only. "
            "Does not finalize public/private split governance."
        ),
    }
    write_json(str(freeze_path), freeze_meta)

    summary = {
        "status": "ok",
        "freeze_id": args.freeze_id,
        "out_dir": str(out_dir),
        "seed_count": len(seeds),
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
        "freeze_meta_path": str(freeze_path),
        "instance_count": len(suite),
        "split_id": args.split,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    if args.instances:
        instances, _ = load_instance_bundle(args.instances)
        if args.instance_index < 0 or args.instance_index >= len(instances):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "instance_index_out_of_range",
                        "instance_count": len(instances),
                        "instance_index": int(args.instance_index),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        instance = instances[args.instance_index]
    else:
        cfg = GeneratorConfig()
        instance, _ = generate_instance(seed=args.seed, cfg=cfg, split_id=args.split)

    if args.agent and args.script:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "mutually_exclusive_inputs",
                    "message": "choose either --agent or --script, not both",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    eval_track = str(args.eval_track)
    tool_allowlist_id = str(args.tool_allowlist_id).strip()
    tool_log_hash = str(args.tool_log_hash).strip()

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

    if eval_track == "EVAL-CB":
        if tool_allowlist_id.lower() != "none" or tool_log_hash:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "track_policy_violation",
                        "message": (
                            "EVAL-CB forbids external tool metadata; "
                            "use tool_allowlist_id=none and empty tool_log_hash"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    if eval_track == "EVAL-TA":
        if not tool_allowlist_id or tool_allowlist_id.lower() == "none":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "track_policy_violation",
                        "message": "EVAL-TA requires a non-empty tool_allowlist_id",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if not tool_log_hash:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "track_policy_violation",
                        "message": "EVAL-TA requires a non-empty tool_log_hash",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    if args.agent:
        agent = make_agent(args.agent)
        agent_id = str(args.agent).strip().lower()
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
        policy = baseline_policy(agent, instance)
        actor = agent.name
    elif args.script:
        actions_by_t = parse_action_script(args.script)
        policy = scripted_policy(actions_by_t)
        actor = "scripted-policy"
    else:
        policy = human_policy(renderer_track=args.renderer_track)
        actor = "human-interactive"

    try:
        result = run_episode(instance, policy, renderer_track=args.renderer_track)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "episode_execution_error",
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = {
        "status": "ok",
        "actor": actor,
        "run_contract": {
            "eval_track": eval_track,
            "renderer_track": args.renderer_track,
            "tool_allowlist_id": tool_allowlist_id or "none",
            "tool_log_hash": tool_log_hash,
        },
        "instance": instance.to_canonical_dict(),
        "episode": result,
    }
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _panel_ids(panel_arg: str) -> list[str]:
    ids: list[str] = []
    for part in panel_arg.split(","):
        token = part.strip().lower()
        if token:
            ids.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _track_for_agent_id(agent_id: str) -> str:
    if agent_id in {"tool", "bl-03", "bl-03-toolplanner"}:
        return "EVAL-TA"
    if agent_id in {"oracle", "bl-04", "bl-04-exactoracle"}:
        return "EVAL-OC"
    return "EVAL-CB"


def _cmd_pilot_campaign(args: argparse.Namespace) -> int:
    freeze_dir = Path(args.freeze_dir)
    bundle_path = freeze_dir / "instance_bundle_v1.json"
    manifest_path = freeze_dir / "split_manifest_v1.json"
    if not bundle_path.exists() or not manifest_path.exists():
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "freeze_artifacts_missing",
                    "freeze_dir": str(freeze_dir),
                    "required": [str(bundle_path), str(manifest_path)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "output_dir_not_empty",
                    "out_dir": str(out_dir),
                    "message": "use --force to overwrite a non-empty output directory",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    instances, bundle_meta = load_instance_bundle(str(bundle_path))
    instance_lookup = {inst.instance_id: inst for inst in instances}
    rows: list[dict[str, object]] = []
    panel = _panel_ids(args.baseline_panel)

    for idx, agent_id in enumerate(panel):
        agent = make_agent(agent_id)
        eval_track = _track_for_agent_id(agent_id)
        if eval_track == "EVAL-TA":
            tool_allowlist_id = args.tool_allowlist_id
            tool_log_hash = args.tool_log_hash or stable_hash_json(
                {
                    "freeze_dir": str(freeze_dir),
                    "agent": agent.name,
                    "seed": int(args.seed + idx),
                    "panel_index": idx,
                }
            )[:16]
        else:
            tool_allowlist_id = "none"
            tool_log_hash = ""

        records, _ = evaluate_suite(
            agent=agent,
            instances=instances,
            eval_track=eval_track,
            renderer_track=args.renderer_track,
            seed=args.seed + idx,
        )
        run_meta = {
            "family_id": bundle_meta.get("family_id", FAMILY_ID),
            "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
            "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
            "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
            "harness_version": HARNESS_VERSION,
            "git_commit": current_git_commit(),
            "config_hash": bundle_meta.get("config_hash", "unknown"),
            "tool_allowlist_id": tool_allowlist_id,
            "tool_log_hash": tool_log_hash,
        }
        for record in records:
            rows.append(
                run_record_to_dict(
                    record,
                    instance=instance_lookup.get(record.instance_id),
                    run_meta=run_meta,
                )
            )

    for external_path in args.external_runs:
        rows.extend(load_jsonl(external_path))

    runs_path = out_dir / "runs_combined.jsonl"
    write_jsonl(str(runs_path), rows)

    err_payload, coverage_payload = _validate_runs_manifest(
        rows,
        manifest_path=str(manifest_path),
        strict_mode=True,
        official_mode=True,
    )
    validation_path = out_dir / "official_validation.json"
    report_path = out_dir / "official_report.json"

    if err_payload is not None:
        write_json(str(validation_path), err_payload)
        print(json.dumps(err_payload, indent=2, sort_keys=True))
        return 2

    validation_ok = {
        "status": "ok",
        "strict_mode": True,
        "official_mode": True,
        "rows": len(rows),
        "manifest_path": str(manifest_path),
        "runs_path": str(runs_path),
    }
    if coverage_payload is not None:
        validation_ok["manifest_coverage"] = coverage_payload
    write_json(str(validation_path), validation_ok)

    report = _build_report_payload(
        rows=rows,
        strict_mode=True,
        coverage_payload=coverage_payload,
    )
    write_json(str(report_path), report)

    summary = {
        "status": "ok",
        "freeze_dir": str(freeze_dir),
        "out_dir": str(out_dir),
        "runs_path": str(runs_path),
        "validation_path": str(validation_path),
        "report_path": str(report_path),
        "row_count": len(rows),
        "baseline_panel": panel,
        "external_runs_count": len(args.external_runs),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GF-01 benchmark harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run a single-instance baseline demo")
    p_demo.add_argument("--seed", type=int, default=1337)
    p_demo.set_defaults(func=_cmd_demo)

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
    p_gen.set_defaults(func=_cmd_generate)

    p_chk = sub.add_parser("checks", help="Run priority H3 checks AB-001/003/005/007/009")
    p_chk.add_argument("--seed", type=int, default=3000)
    p_chk.set_defaults(func=_cmd_checks)

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
    p_prof.set_defaults(func=_cmd_profile)

    p_eval = sub.add_parser("evaluate", help="Evaluate one baseline agent on external instances")
    p_eval.add_argument("--instances", type=str, required=True, help="Path to instance JSON")
    p_eval.add_argument("--agent", type=str, default="greedy", help="random|greedy|search|tool|oracle")
    p_eval.add_argument("--eval-track", type=str, default="EVAL-CB")
    p_eval.add_argument("--renderer-track", type=str, default="json")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--tool-allowlist-id", type=str, default="none")
    p_eval.add_argument("--tool-log-hash", type=str, default="")
    p_eval.add_argument("--out", type=str, default="", help="Optional output JSONL path")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_report = sub.add_parser("report", help="Aggregate run JSONL by track/split/mode")
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
    p_report.set_defaults(func=_cmd_report)

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
    p_validate.set_defaults(func=_cmd_validate)

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
    p_migrate.add_argument("--default-eval-track", type=str, default="EVAL-CB")
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
    p_migrate.set_defaults(func=_cmd_migrate_runs)

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
    p_gate.set_defaults(func=_cmd_gate)

    p_manifest = sub.add_parser("manifest", help="Build split manifest from an instance file")
    p_manifest.add_argument("--instances", type=str, required=True, help="Path to instance JSON")
    p_manifest.add_argument("--out", type=str, default="", help="Optional output manifest path")
    p_manifest.set_defaults(func=_cmd_manifest)

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
    p_freeze.add_argument("--out-dir", type=str, default="pilot_freeze/gf01_pilot_freeze_v1")
    p_freeze.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_freeze.set_defaults(func=_cmd_freeze_pilot)

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
    p_campaign.add_argument("--renderer-track", type=str, default="json", choices=["json", "visual"])
    p_campaign.add_argument("--seed", type=int, default=1100)
    p_campaign.add_argument("--tool-allowlist-id", type=str, default="local-planner-v1")
    p_campaign.add_argument("--tool-log-hash", type=str, default="")
    p_campaign.add_argument(
        "--external-runs",
        action="append",
        default=[],
        help="Optional path to external run JSONL (repeatable)",
    )
    p_campaign.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_campaign.set_defaults(func=_cmd_pilot_campaign)

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
    p_play.add_argument("--renderer-track", type=str, choices=["json", "visual"], default="visual")
    p_play.add_argument("--eval-track", type=str, default="EVAL-CB", choices=list(ALLOWED_EVAL_TRACKS))
    p_play.add_argument("--tool-allowlist-id", type=str, default="none")
    p_play.add_argument("--tool-log-hash", type=str, default="")
    p_play.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_play.set_defaults(func=_cmd_play)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
