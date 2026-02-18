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
from collections import defaultdict
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
    ADAPTATION_POLICY_VERSION,
    ALLOWED_EVAL_TRACKS,
    ALLOWED_ADAPTATION_CONDITIONS,
    ALLOWED_ADAPTATION_DATA_SCOPES,
    ALLOWED_MODES,
    ALLOWED_PLAY_PROTOCOLS,
    ALLOWED_TOOL_ALLOWLISTS_BY_TRACK,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    DEFAULT_PRIVATE_EVAL_MIN_COUNT,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_RATIO_TOLERANCE,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    INSTANCE_BUNDLE_SCHEMA_VERSION,
    OFFICIAL_SPLITS,
    PILOT_FREEZE_SCHEMA_VERSION,
    RUN_RECORD_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
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


def _track_tool_policy_message(
    *,
    eval_track: str,
    tool_allowlist_id: str,
    tool_log_hash: str,
) -> str | None:
    allowlist = str(tool_allowlist_id).strip()
    tool_hash = str(tool_log_hash).strip()
    if eval_track not in ALLOWED_TOOL_ALLOWLISTS_BY_TRACK:
        return f"unsupported eval_track {eval_track}"
    if eval_track == "EVAL-CB":
        if allowlist.lower() != "none" or tool_hash:
            return (
                "EVAL-CB forbids external tool metadata; "
                "use tool_allowlist_id=none and empty tool_log_hash"
            )
        return None
    allowed = ALLOWED_TOOL_ALLOWLISTS_BY_TRACK[eval_track]
    if allowlist not in allowed:
        return (
            f"{eval_track} requires tool_allowlist_id in {list(allowed)} "
            f"(received {allowlist or '<empty>'})"
        )
    if not tool_hash:
        return f"{eval_track} requires a non-empty tool_log_hash"
    if tool_hash.lower() == "unknown":
        return f"{eval_track} requires tool_log_hash != unknown"
    return None


def _adaptation_policy_message(
    *,
    adaptation_condition: str,
    adaptation_budget_tokens: int,
    adaptation_data_scope: str,
    adaptation_protocol_id: str,
) -> str | None:
    if adaptation_condition not in ALLOWED_ADAPTATION_CONDITIONS:
        return (
            f"unsupported adaptation_condition {adaptation_condition}; "
            f"expected one of {list(ALLOWED_ADAPTATION_CONDITIONS)}"
        )
    if adaptation_data_scope not in ALLOWED_ADAPTATION_DATA_SCOPES:
        return (
            f"unsupported adaptation_data_scope {adaptation_data_scope}; "
            f"expected one of {list(ALLOWED_ADAPTATION_DATA_SCOPES)}"
        )
    if adaptation_budget_tokens < 0:
        return "adaptation_budget_tokens must be >= 0"
    protocol = str(adaptation_protocol_id).strip()
    if adaptation_condition == "no_adaptation":
        if adaptation_budget_tokens != 0:
            return "no_adaptation requires adaptation_budget_tokens=0"
        if adaptation_data_scope != "none":
            return "no_adaptation requires adaptation_data_scope=none"
        if protocol not in {"", "none"}:
            return "no_adaptation requires adaptation_protocol_id=none"
        return None
    if adaptation_budget_tokens <= 0:
        return f"{adaptation_condition} requires adaptation_budget_tokens>0"
    if adaptation_data_scope == "none":
        return f"{adaptation_condition} requires adaptation_data_scope!=none"
    if not protocol or protocol.lower() in {"none", "unknown"}:
        return f"{adaptation_condition} requires non-empty adaptation_protocol_id"
    return None


def _parse_split_ratio_arg(text: str) -> dict[str, float]:
    ratio_map: dict[str, float] = {}
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"invalid ratio token '{token}' (expected split=value)")
        split_id, value_text = token.split("=", 1)
        split_key = split_id.strip()
        if not split_key:
            raise ValueError(f"invalid split id in token '{token}'")
        value = float(value_text.strip())
        if value < 0.0:
            raise ValueError(f"ratio for {split_key} must be non-negative")
        ratio_map[split_key] = value
    if not ratio_map:
        raise ValueError("split ratio map is empty")
    total = sum(ratio_map.values())
    if total <= 0:
        raise ValueError("split ratio map must sum to a positive value")
    return {k: v / total for k, v in ratio_map.items()}


def _split_policy_report(
    manifest: dict[str, object],
    *,
    target_ratios: dict[str, float],
    tolerance: float,
    private_split_id: str,
    min_private_eval_count: int,
    require_official_split_names: bool,
) -> tuple[dict[str, object], bool]:
    entries = manifest.get("instances", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.instances must be a list")
    total = len(entries)
    counts: dict[str, int] = defaultdict(int)
    invalid_split_entries: list[dict[str, object]] = []
    allowed_splits = set(OFFICIAL_SPLITS)

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            invalid_split_entries.append({"index": idx, "reason": "entry_not_object"})
            continue
        split_id = str(entry.get("split_id", "")).strip()
        if not split_id:
            invalid_split_entries.append({"index": idx, "reason": "missing_split_id"})
            continue
        counts[split_id] += 1
        if require_official_split_names and split_id not in allowed_splits:
            invalid_split_entries.append(
                {"index": idx, "reason": f"unsupported_split_id:{split_id}"}
            )

    observed_ratios = {
        split_id: (count / float(total) if total > 0 else 0.0)
        for split_id, count in sorted(counts.items())
    }
    target_rows = []
    ratio_failures: list[dict[str, object]] = []
    for split_id in sorted(target_ratios):
        target = float(target_ratios[split_id])
        observed = observed_ratios.get(split_id, 0.0)
        delta = abs(observed - target)
        row = {
            "split_id": split_id,
            "target_ratio": target,
            "observed_ratio": observed,
            "delta": delta,
            "within_tolerance": delta <= tolerance,
            "count": int(counts.get(split_id, 0)),
        }
        target_rows.append(row)
        if delta > tolerance:
            ratio_failures.append(row)

    private_count = int(counts.get(private_split_id, 0))
    private_min_pass = private_count >= int(min_private_eval_count)
    no_invalid_splits = len(invalid_split_entries) == 0
    ratio_pass = len(ratio_failures) == 0
    passed = bool(ratio_pass and no_invalid_splits and private_min_pass)

    report = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "split_policy_violation",
        "split_policy_version": SPLIT_POLICY_VERSION,
        "manifest_schema_version": manifest.get("schema_version", "unknown"),
        "instance_count": total,
        "target_ratios": target_ratios,
        "observed_counts": dict(sorted(counts.items())),
        "observed_ratios": observed_ratios,
        "tolerance": float(tolerance),
        "ratio_checks": target_rows,
        "ratio_failures": ratio_failures,
        "require_official_split_names": bool(require_official_split_names),
        "invalid_split_entries_preview": invalid_split_entries[:20],
        "private_split_id": private_split_id,
        "private_eval_count": private_count,
        "private_eval_min_count": int(min_private_eval_count),
        "private_min_pass": private_min_pass,
    }
    return report, passed


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
    eval_track = str(args.eval_track).strip()
    renderer_track = args.renderer_track
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

    policy_msg = _track_tool_policy_message(
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

    adaptation_msg = _adaptation_policy_message(
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
        "git_commit": current_git_commit(),
        "config_hash": bundle_meta.get("config_hash", "unknown"),
        "tool_allowlist_id": tool_allowlist_id or "none",
        "tool_log_hash": tool_log_hash,
        "play_protocol": "commit_only",
        "scored_commit_episode": True,
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
        "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
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
    groups: dict[
        tuple[str, str, str, str, str, bool, str, int, str, str],
        list[dict[str, object]],
    ] = {}
    for row in rows:
        try:
            adaptation_budget_tokens = int(row.get("adaptation_budget_tokens", 0))
        except Exception:
            adaptation_budget_tokens = -1
        key = (
            str(row.get("eval_track", "unknown")),
            str(row.get("renderer_track", "unknown")),
            str(row.get("split_id", "unknown")),
            str(row.get("mode", "unknown")),
            str(row.get("play_protocol", "unknown")),
            bool(row.get("scored_commit_episode", True)),
            str(row.get("adaptation_condition", "unknown")),
            adaptation_budget_tokens,
            str(row.get("adaptation_data_scope", "unknown")),
            str(row.get("adaptation_protocol_id", "unknown")),
        )
        groups.setdefault(key, []).append(row)

    report_rows = []
    for key, entries in sorted(groups.items()):
        (
            eval_track,
            renderer_track,
            split_id,
            mode,
            play_protocol,
            scored_commit_episode,
            adaptation_condition,
            adaptation_budget_tokens,
            adaptation_data_scope,
            adaptation_protocol_id,
        ) = key
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
                "play_protocol": play_protocol,
                "scored_commit_episode": scored_commit_episode,
                "adaptation_condition": adaptation_condition,
                "adaptation_budget_tokens": adaptation_budget_tokens,
                "adaptation_data_scope": adaptation_data_scope,
                "adaptation_protocol_id": adaptation_protocol_id,
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
    default_eval_track = str(args.default_eval_track).strip()
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

    policy_msg = _track_tool_policy_message(
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
    adaptation_msg = _adaptation_policy_message(
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
        "git_commit": args.git_commit or current_git_commit(),
        "config_hash": args.config_hash,
        "tool_allowlist_id": tool_allowlist_id,
        "tool_log_hash": tool_log_hash,
        "play_protocol": "commit_only",
        "scored_commit_episode": True,
        "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
        "eval_track": default_eval_track,
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


def _cmd_split_policy_check(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    manifest_errors = validate_manifest(manifest, strict=bool(args.strict_manifest))
    if manifest_errors:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "manifest_schema_validation",
                    "strict_manifest": bool(args.strict_manifest),
                    "error_count": len(manifest_errors),
                    "errors_preview": manifest_errors[:50],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        target_ratios = _parse_split_ratio_arg(args.target_ratios)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "invalid_split_ratio_arg",
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        report, passed = _split_policy_report(
            manifest,
            target_ratios=target_ratios,
            tolerance=float(args.tolerance),
            private_split_id=str(args.private_split),
            min_private_eval_count=int(args.min_private_eval_count),
            require_official_split_names=bool(args.require_official_split_names),
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "split_policy_check_error",
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


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
    mode_override = str(args.mode).strip() if args.mode else ""
    if mode_override and mode_override not in ALLOWED_MODES:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "unsupported_mode",
                    "message": f"mode must be one of {list(ALLOWED_MODES)}",
                    "mode": mode_override,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

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

    suite = generate_suite(
        seeds=seeds,
        split_id=args.split,
        cfg=cfg,
        mode=mode_override or None,
    )
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
        "mode_override": mode_override or "mixed",
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
        "mode_override": mode_override or "mixed",
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
        "mode_override": mode_override or "mixed",
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

    eval_track = str(args.eval_track).strip()
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

    policy_msg = _track_tool_policy_message(
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

    adaptation_msg = _adaptation_policy_message(
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
            "play_protocol": "commit_only",
            "scored_commit_episode": True,
            "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
            "adaptation_condition": adaptation_condition,
            "adaptation_budget_tokens": adaptation_budget_tokens,
            "adaptation_data_scope": adaptation_data_scope,
            "adaptation_protocol_id": adaptation_protocol_id or "none",
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
    adaptation_condition = str(args.adaptation_condition).strip()
    adaptation_budget_tokens = int(args.adaptation_budget_tokens)
    adaptation_data_scope = str(args.adaptation_data_scope).strip()
    adaptation_protocol_id = str(args.adaptation_protocol_id).strip()
    adaptation_msg = _adaptation_policy_message(
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
        elif eval_track == "EVAL-OC":
            tool_allowlist_id = DEFAULT_TOOL_ALLOWLIST_BY_TRACK["EVAL-OC"]
            tool_log_hash = stable_hash_json(
                {
                    "freeze_dir": str(freeze_dir),
                    "agent": agent.name,
                    "seed": int(args.seed + idx),
                    "panel_index": idx,
                    "policy": "oracle-ceiling",
                }
            )[:16]
        else:
            tool_allowlist_id = "none"
            tool_log_hash = ""

        policy_msg = _track_tool_policy_message(
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
                        "agent": agent.name,
                        "eval_track": eval_track,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

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
            "play_protocol": "commit_only",
            "scored_commit_episode": True,
            "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
            "adaptation_condition": adaptation_condition,
            "adaptation_budget_tokens": adaptation_budget_tokens,
            "adaptation_data_scope": adaptation_data_scope,
            "adaptation_protocol_id": adaptation_protocol_id or "none",
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
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _analysis_rate(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(field, False))) / float(len(rows))


def _complexity_score(row: dict[str, object]) -> float | None:
    complexity = row.get("complexity")
    if not isinstance(complexity, dict) or not complexity:
        return None
    values: list[float] = []
    for value in complexity.values():
        try:
            values.append(float(value))
        except Exception:
            continue
    if not values:
        return None
    return statistics.fmean(values)


def _assign_complexity_quartiles(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for row in rows:
        score = _complexity_score(row)
        if score is None:
            continue
        cloned = dict(row)
        cloned["complexity_score"] = score
        scored.append(cloned)
    scored.sort(key=lambda item: float(item["complexity_score"]))
    n = len(scored)
    if n == 0:
        return []
    for idx, row in enumerate(scored):
        quartile_idx = min(4, (idx * 4) // n + 1)
        row["quartile"] = f"Q{quartile_idx}"
    return scored


def _agent_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str, str, bool, str, int, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in rows:
        try:
            adaptation_budget_tokens = int(row.get("adaptation_budget_tokens", 0))
        except Exception:
            adaptation_budget_tokens = -1
        key = (
            str(row.get("agent_name", "unknown")),
            str(row.get("eval_track", "unknown")),
            str(row.get("mode", "unknown")),
            str(row.get("play_protocol", "unknown")),
            bool(row.get("scored_commit_episode", True)),
            str(row.get("adaptation_condition", "unknown")),
            adaptation_budget_tokens,
            str(row.get("adaptation_data_scope", "unknown")),
            str(row.get("adaptation_protocol_id", "unknown")),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        (
            agent_name,
            eval_track,
            mode,
            play_protocol,
            scored_commit_episode,
            adaptation_condition,
            adaptation_budget_tokens,
            adaptation_data_scope,
            adaptation_protocol_id,
        ) = key
        entries = grouped[key]
        summary_rows.append(
            {
                "agent_name": agent_name,
                "eval_track": eval_track,
                "mode": mode,
                "play_protocol": play_protocol,
                "scored_commit_episode": scored_commit_episode,
                "adaptation_condition": adaptation_condition,
                "adaptation_budget_tokens": adaptation_budget_tokens,
                "adaptation_data_scope": adaptation_data_scope,
                "adaptation_protocol_id": adaptation_protocol_id,
                "count": len(entries),
                "goal_rate": _analysis_rate(entries, "goal"),
                "certified_rate": _analysis_rate(entries, "valid"),
                "ap_f1_mean": statistics.fmean(float(e.get("ap_f1", 0.0)) for e in entries),
                "ts_f1_mean": statistics.fmean(float(e.get("ts_f1", 0.0)) for e in entries),
            }
        )
    return summary_rows


def _cmd_pilot_analyze(args: argparse.Namespace) -> int:
    campaign_dir = Path(args.campaign_dir)
    runs_path = campaign_dir / "runs_combined.jsonl"
    validation_path = campaign_dir / "official_validation.json"

    if not runs_path.exists():
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "campaign_runs_missing",
                    "campaign_dir": str(campaign_dir),
                    "runs_path": str(runs_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    rows = load_jsonl(str(runs_path))
    schema_errors = validate_run_rows(rows, strict=True)
    if schema_errors:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "run_schema_validation",
                    "strict_mode": True,
                    "error_count": len(schema_errors),
                    "errors_preview": schema_errors[:50],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    validation_payload: dict[str, object] | None = None
    if validation_path.exists():
        validation_payload = load_json(str(validation_path))

    eval_track = str(args.eval_track)
    mode = str(args.mode)
    greedy_agent_name = str(args.greedy_agent_name)
    public_splits = {token.strip() for token in str(args.public_splits).split(",") if token.strip()}

    mode_rows = [
        row
        for row in rows
        if str(row.get("eval_track", "")) == eval_track
        and str(row.get("mode", "")) == mode
        and bool(row.get("scored_commit_episode", True))
    ]
    greedy_rows = [row for row in mode_rows if str(row.get("agent_name", "")) == greedy_agent_name]
    held_out_greedy_rows = [
        row for row in greedy_rows if str(row.get("split_id", "")) not in public_splits
    ]
    if not held_out_greedy_rows:
        held_out_greedy_rows = greedy_rows

    quartiled = _assign_complexity_quartiles(held_out_greedy_rows)
    quartile_rows: dict[str, list[dict[str, object]]] = {q: [] for q in ("Q1", "Q2", "Q3", "Q4")}
    for row in quartiled:
        quartile = str(row.get("quartile", "Q4"))
        quartile_rows.setdefault(quartile, []).append(row)

    quartile_stats: list[dict[str, object]] = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        bucket = quartile_rows.get(quartile, [])
        stats_row: dict[str, object] = {
            "quartile": quartile,
            "count": len(bucket),
            "certified_rate": _analysis_rate(bucket, "valid"),
            "goal_rate": _analysis_rate(bucket, "goal"),
        }
        if bucket:
            scores = [float(item.get("complexity_score", 0.0)) for item in bucket]
            stats_row["complexity_score_min"] = min(scores)
            stats_row["complexity_score_max"] = max(scores)
        quartile_stats.append(stats_row)

    q1_stats = quartile_stats[0]
    q4_stats = quartile_stats[3]
    m_q1 = float(q1_stats["certified_rate"])
    m_q4 = float(q4_stats["certified_rate"])
    discrimination_delta = m_q1 - m_q4
    discrimination_trigger = (
        discrimination_delta < float(args.discrimination_delta_threshold)
        or m_q4 < float(args.discrimination_q4_floor)
    )

    greedy_goal = _analysis_rate(held_out_greedy_rows, "goal")
    greedy_m = _analysis_rate(held_out_greedy_rows, "valid")
    shortcut_trigger = (
        greedy_goal > float(args.shortcut_goal_threshold)
        and greedy_m < float(args.shortcut_certified_floor)
    )

    normal_unique_instance_count = len(
        {
            str(row.get("instance_id"))
            for row in mode_rows
            if str(row.get("instance_id", "")).strip()
        }
    )
    sample_trigger_reached = normal_unique_instance_count >= int(args.sample_target)

    oracle_rows = [
        row
        for row in rows
        if str(row.get("mode", "")) == mode
        and str(row.get("eval_track", "")) == "EVAL-OC"
        and "oracle" in str(row.get("agent_name", "")).lower()
    ]
    oracle_minus_greedy = _analysis_rate(oracle_rows, "valid") - greedy_m if oracle_rows else None

    recommendation = "keep_coefficients"
    if discrimination_trigger or shortcut_trigger:
        recommendation = "recalibrate_normal_window"

    out = {
        "status": "ok",
        "policy_reference": "DEC-014d",
        "campaign_dir": str(campaign_dir),
        "runs_path": str(runs_path),
        "validation_path": str(validation_path) if validation_path.exists() else "",
        "row_count": len(rows),
        "analysis_scope": {
            "eval_track": eval_track,
            "mode": mode,
            "greedy_agent_name": greedy_agent_name,
            "public_splits": sorted(public_splits),
            "held_out_fallback_used": not bool(
                [
                    row
                    for row in greedy_rows
                    if str(row.get("split_id", "")) not in public_splits
                ]
            ),
        },
        "thresholds": {
            "sample_target": int(args.sample_target),
            "discrimination_delta_threshold": float(args.discrimination_delta_threshold),
            "discrimination_q4_floor": float(args.discrimination_q4_floor),
            "shortcut_goal_threshold": float(args.shortcut_goal_threshold),
            "shortcut_certified_floor": float(args.shortcut_certified_floor),
        },
        "sample_progress": {
            "normal_unique_instance_count": normal_unique_instance_count,
            "sample_trigger_reached": sample_trigger_reached,
        },
        "discrimination_check": {
            "m_q1": m_q1,
            "m_q4": m_q4,
            "m_q1_minus_m_q4": discrimination_delta,
            "quartile_stats": quartile_stats,
            "triggered": discrimination_trigger,
        },
        "shortcut_check": {
            "goal_rate": greedy_goal,
            "certified_rate": greedy_m,
            "triggered": shortcut_trigger,
            "row_count": len(held_out_greedy_rows),
        },
        "oracle_context": {
            "oracle_minus_greedy_certified_rate": oracle_minus_greedy,
            "oracle_row_count": len(oracle_rows),
        },
        "agent_summary": _agent_summary_rows(rows),
        "recommendation": recommendation,
    }
    if validation_payload is not None:
        out["official_validation"] = validation_payload

    out_path = Path(args.out) if args.out else campaign_dir / "pilot_analysis.json"
    write_json(str(out_path), out)
    out["out_path"] = str(out_path)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GF-01 benchmark harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    default_ratio_arg = ",".join(
        f"{split_id}={DEFAULT_SPLIT_RATIOS[split_id]}" for split_id in OFFICIAL_SPLITS
    )

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
    p_eval.set_defaults(func=_cmd_evaluate)

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
    p_split_policy.set_defaults(func=_cmd_split_policy_check)

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
    p_campaign.set_defaults(func=_cmd_pilot_campaign)

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
    p_analysis.set_defaults(func=_cmd_pilot_analyze)

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
    p_play.set_defaults(func=_cmd_play)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
