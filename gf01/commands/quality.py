"""Quality, governance, profiling, and release-candidate commands."""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

from ..checks import run_priority_h3_checks
from ..gate import run_regression_gate
from ..identifiability import (
    identifiability_policy_error,
    instance_identifiability_metrics,
)
from ..io import (
    load_instance_bundle,
    load_json,
    load_jsonl,
    validate_manifest,
    write_json,
)
from ..meta import (
    ADAPTATION_POLICY_VERSION,
    BASELINE_PANEL_CORE,
    BASELINE_PANEL_FULL,
    BASELINE_PANEL_POLICY_VERSION,
    CHECKER_VERSION,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    IDENTIFIABILITY_METRIC_ID,
    IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
    IDENTIFIABILITY_POLICY_VERSION,
    RENDERER_POLICY_VERSION,
    ROTATION_POLICY_VERSION,
    RUN_RECORD_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    current_git_commit,
    renderer_profile_for_track,
)
from ..models import GeneratorConfig
from ..profiling import profile_pipeline
from . import pilot as pilot_commands
from .shared import (
    canonical_baseline_agent_id,
    parse_split_ratio_arg,
    release_rotation_report,
    split_policy_report,
    track_for_agent_id,
    validate_runs_manifest,
)


def cmd_checks(args: argparse.Namespace) -> int:
    results = run_priority_h3_checks(seed_base=args.seed)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
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


def cmd_gate(args: argparse.Namespace) -> int:
    if int(args.unittest_shards) < 1:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "invalid_unittest_shards",
                    "message": "--unittest-shards must be >= 1",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    summary = run_regression_gate(
        root=Path(".").resolve(),
        fixture_root=Path(args.fixture_root),
        seed_checks=int(args.seed_checks),
        seed_profile=int(args.seed_profile),
        public_count=int(args.public_count),
        private_count=int(args.private_count),
        unittest_shards=int(args.unittest_shards),
        fail_fast=bool(args.fail_fast),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary.get("passed", False)) else 2


def cmd_split_policy_check(args: argparse.Namespace) -> int:
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
        target_ratios = parse_split_ratio_arg(args.target_ratios)
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
        report, passed = split_policy_report(
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


def cmd_release_governance_check(args: argparse.Namespace) -> int:
    min_public_novelty_ratio = float(args.min_public_novelty_ratio)
    if min_public_novelty_ratio < 0.0 or min_public_novelty_ratio > 1.0:
        out = {
            "status": "error",
            "error_type": "invalid_min_public_novelty_ratio",
            "message": "min_public_novelty_ratio must be in [0, 1]",
        }
        if args.out:
            write_json(args.out, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2

    strict_manifest = bool(getattr(args, "strict_manifest", False)) and not bool(
        getattr(args, "no_strict_manifest", False)
    )
    require_official_split_names = bool(
        getattr(args, "require_official_split_names", False)
    ) and not bool(getattr(args, "allow_non_official_split_names", False))

    current_manifest = load_json(args.manifest)
    manifest_errors = validate_manifest(current_manifest, strict=strict_manifest)
    if manifest_errors:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "manifest_schema_validation",
                    "strict_manifest": strict_manifest,
                    "error_count": len(manifest_errors),
                    "errors_preview": manifest_errors[:50],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    previous_manifest: dict[str, object] | None = None
    if args.previous_manifest:
        previous_manifest = load_json(args.previous_manifest)
        prev_errors = validate_manifest(previous_manifest, strict=strict_manifest)
        if prev_errors:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "previous_manifest_schema_validation",
                        "strict_manifest": strict_manifest,
                        "error_count": len(prev_errors),
                        "errors_preview": prev_errors[:50],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

    try:
        target_ratios = parse_split_ratio_arg(args.target_ratios)
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
        split_report, split_pass = split_policy_report(
            current_manifest,
            target_ratios=target_ratios,
            tolerance=float(args.tolerance),
            private_split_id=str(args.private_split),
            min_private_eval_count=int(args.min_private_eval_count),
            require_official_split_names=require_official_split_names,
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

    if bool(args.require_previous_manifest) and previous_manifest is None:
        out = {
            "status": "error",
            "error_type": "release_rotation_previous_manifest_required",
            "message": "set --previous-manifest when --require-previous-manifest is enabled",
            "rotation_policy_version": ROTATION_POLICY_VERSION,
        }
        if args.out:
            write_json(args.out, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2

    try:
        rotation_report, rotation_pass = release_rotation_report(
            current_manifest=current_manifest,
            previous_manifest=previous_manifest,
            private_split_id=str(args.private_split),
            min_public_novelty_ratio=min_public_novelty_ratio,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "release_rotation_check_error",
                    "message": str(exc),
                    "rotation_policy_version": ROTATION_POLICY_VERSION,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    passed = bool(split_pass and rotation_pass)
    report = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "release_governance_violation",
        "split_policy_version": SPLIT_POLICY_VERSION,
        "rotation_policy_version": ROTATION_POLICY_VERSION,
        "manifest_path": str(args.manifest),
        "previous_manifest_path": str(args.previous_manifest or ""),
        "strict_manifest": strict_manifest,
        "require_official_split_names": require_official_split_names,
        "split_policy": split_report,
        "rotation_policy": rotation_report,
        "passed": passed,
    }
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def cmd_release_report_check(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.runs)
    err_payload, coverage_payload = validate_runs_manifest(
        rows,
        manifest_path=args.manifest,
        strict_mode=True,
        official_mode=True,
    )
    if err_payload is not None:
        print(json.dumps(err_payload, indent=2, sort_keys=True))
        return 2
    manifest = load_json(args.manifest)

    baseline_policy_level = str(args.baseline_policy_level).strip()
    required_agents = (
        list(BASELINE_PANEL_FULL)
        if baseline_policy_level == "full"
        else list(BASELINE_PANEL_CORE)
    )
    required_tracks = sorted({track_for_agent_id(agent_id) for agent_id in required_agents})

    expected_instances_by_slice: dict[tuple[str, str], set[str]] = defaultdict(set)
    entries = manifest.get("instances", [])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            split_id = str(entry.get("split_id", "")).strip()
            mode = str(entry.get("mode", "")).strip()
            instance_id = str(entry.get("instance_id", "")).strip()
            if split_id and mode and instance_id:
                expected_instances_by_slice[(split_id, mode)].add(instance_id)

    agent_slice_coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    track_slice_coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    seen_required_agents: set[str] = set()

    for row in rows:
        split_id = str(row.get("split_id", "")).strip()
        mode = str(row.get("mode", "")).strip()
        instance_id = str(row.get("instance_id", "")).strip()
        eval_track = str(row.get("eval_track", "")).strip()
        if not split_id or not mode or not instance_id:
            continue

        agent_name = str(row.get("agent_name", "")).strip()
        try:
            canonical_agent = canonical_baseline_agent_id(agent_name)
        except ValueError:
            canonical_agent = ""
        if canonical_agent not in required_agents:
            continue
        seen_required_agents.add(canonical_agent)
        agent_slice_coverage[(canonical_agent, split_id, mode)].add(instance_id)
        if eval_track in required_tracks:
            track_slice_coverage[(eval_track, split_id, mode)].add(instance_id)

    missing_agent_ids = sorted(set(required_agents) - seen_required_agents)
    missing_agent_slice: list[dict[str, object]] = []
    missing_track_slice: list[dict[str, object]] = []
    for (split_id, mode), expected_instances in sorted(expected_instances_by_slice.items()):
        expected_count = len(expected_instances)
        for agent_id in required_agents:
            covered_instances = agent_slice_coverage.get((agent_id, split_id, mode), set())
            if len(covered_instances) < expected_count:
                missing_agent_slice.append(
                    {
                        "agent_id": agent_id,
                        "split_id": split_id,
                        "mode": mode,
                        "expected_count": expected_count,
                        "covered_count": len(covered_instances),
                    }
                )
        for eval_track in required_tracks:
            covered_instances = track_slice_coverage.get((eval_track, split_id, mode), set())
            if len(covered_instances) < expected_count:
                missing_track_slice.append(
                    {
                        "eval_track": eval_track,
                        "split_id": split_id,
                        "mode": mode,
                        "expected_count": expected_count,
                        "covered_count": len(covered_instances),
                    }
                )

    passed = not missing_agent_ids and not missing_agent_slice and not missing_track_slice
    out = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "release_report_policy_violation",
        "baseline_policy_version": BASELINE_PANEL_POLICY_VERSION,
        "baseline_policy_level": baseline_policy_level,
        "required_agents": required_agents,
        "required_eval_tracks": required_tracks,
        "missing_required_agents": missing_agent_ids,
        "missing_agent_slice_requirements_preview": missing_agent_slice[:50],
        "missing_track_slice_requirements_preview": missing_track_slice[:50],
        "manifest_coverage": coverage_payload or {},
        "passed": passed,
    }
    if args.out:
        write_json(args.out, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if passed else 2


def cmd_identifiability_check(args: argparse.Namespace) -> int:
    instances, bundle_meta = load_instance_bundle(args.instances)
    min_response_ratio = float(args.min_response_ratio)
    min_unique_signatures = int(args.min_unique_signatures)

    if not instances:
        out = {
            "status": "error",
            "error_type": "empty_instance_set",
            "message": "identifiability check requires at least one instance",
            "instances_path": args.instances,
        }
        if args.out:
            write_json(args.out, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    stored_field_mismatch_count = 0
    for instance in instances:
        computed = instance_identifiability_metrics(instance)
        policy_error = identifiability_policy_error(
            computed,
            min_response_ratio=min_response_ratio,
            min_unique_signatures=min_unique_signatures,
        )
        stored = dict(instance.identifiability) if instance.identifiability else {}

        stored_mismatch = False
        if stored:
            for key in (
                "response_ratio",
                "unique_signature_count",
                "candidate_atom_count",
                "response_atom_count",
            ):
                if key in stored and stored.get(key) != computed.get(key):
                    stored_mismatch = True
                    break
        if stored_mismatch:
            stored_field_mismatch_count += 1

        row = {
            "instance_id": instance.instance_id,
            "passes_threshold": policy_error is None,
            "policy_error": policy_error or "",
            "computed": computed,
            "stored_available": bool(stored),
            "stored_mismatch": bool(stored_mismatch),
        }
        rows.append(row)
        if policy_error is not None:
            failures.append(
                {
                    "instance_id": instance.instance_id,
                    "policy_error": policy_error,
                    "response_ratio": float(computed.get("response_ratio", 0.0)),
                    "unique_signature_count": int(computed.get("unique_signature_count", 0)),
                }
            )

    pass_count = len(instances) - len(failures)
    passed = len(failures) == 0
    out = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "identifiability_policy_violation",
        "policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "metric_id": IDENTIFIABILITY_METRIC_ID,
        "instances_path": args.instances,
        "instance_count": len(instances),
        "pass_count": pass_count,
        "fail_count": len(failures),
        "min_response_ratio": min_response_ratio,
        "min_unique_signatures": min_unique_signatures,
        "bundle_policy_version": bundle_meta.get("identifiability_policy_version", ""),
        "bundle_metric_id": bundle_meta.get("identifiability_metric_id", ""),
        "bundle_min_response_ratio": bundle_meta.get("identifiability_min_response_ratio", None),
        "bundle_min_unique_signatures": bundle_meta.get(
            "identifiability_min_unique_signatures", None
        ),
        "stored_field_mismatch_count": stored_field_mismatch_count,
        "failures_preview": failures[:50],
        "rows": rows,
    }
    if args.out:
        write_json(args.out, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if passed else 2


def invoke_subcommand_silently(
    func,
    args: argparse.Namespace,
) -> tuple[int, dict[str, object], str]:
    """Run a CLI subcommand while capturing JSON stdout for composition."""
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        code = int(func(args))
    stdout_text = capture.getvalue().strip()
    payload: dict[str, object] = {}
    if stdout_text:
        try:
            maybe_payload = json.loads(stdout_text)
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
            else:
                payload = {
                    "status": "error",
                    "error_type": "subcommand_non_object_json",
                    "stdout_preview": str(maybe_payload)[:1000],
                }
        except Exception:
            payload = {
                "status": "error",
                "error_type": "subcommand_non_json_stdout",
                "stdout_preview": stdout_text[:1000],
            }
    return code, payload, stdout_text


def cmd_release_candidate_check(args: argparse.Namespace) -> int:
    freeze_dir = Path(args.freeze_dir)
    campaign_dir = Path(args.campaign_dir)
    manifest_path = freeze_dir / "split_manifest_v1.json"
    runs_path = campaign_dir / "runs_combined.jsonl"

    missing = []
    if not manifest_path.exists():
        missing.append(str(manifest_path))
    if not runs_path.exists():
        missing.append(str(runs_path))
    if missing:
        out = {
            "status": "error",
            "error_type": "release_candidate_source_missing",
            "missing_paths": missing,
        }
        if args.out:
            write_json(args.out, out)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 2

    gov_args = argparse.Namespace(
        manifest=str(manifest_path),
        previous_manifest=str(args.previous_manifest),
        require_previous_manifest=bool(args.require_previous_manifest),
        target_ratios=str(args.target_ratios),
        tolerance=float(args.tolerance),
        private_split=str(args.private_split),
        min_private_eval_count=int(args.min_private_eval_count),
        min_public_novelty_ratio=float(args.min_public_novelty_ratio),
        allow_non_official_split_names=bool(args.allow_non_official_split_names),
        no_strict_manifest=bool(args.no_strict_manifest),
        strict_manifest=True,
        require_official_split_names=True,
        out="",
    )
    gov_code, gov_payload, _ = invoke_subcommand_silently(
        cmd_release_governance_check,
        gov_args,
    )

    report_args = argparse.Namespace(
        runs=str(runs_path),
        manifest=str(manifest_path),
        baseline_policy_level=str(args.baseline_policy_level),
        out="",
    )
    report_code, report_payload, _ = invoke_subcommand_silently(
        cmd_release_report_check,
        report_args,
    )

    package_payload: dict[str, object] = {}
    package_code = 0
    package_skipped = bool(args.skip_package)
    package_out_dir = str(args.package_out_dir)
    if package_skipped:
        package_payload = {
            "status": "skipped",
            "reason": "skip_package_enabled",
        }
    else:
        package_args = argparse.Namespace(
            freeze_dir=str(freeze_dir),
            campaign_dir=str(campaign_dir),
            out_dir=package_out_dir,
            force=bool(args.force_package),
        )
        package_code, package_payload, _ = invoke_subcommand_silently(
            pilot_commands.cmd_release_package,
            package_args,
        )

    gov_passed = bool(gov_code == 0 and gov_payload.get("passed", False))
    report_passed = bool(report_code == 0 and report_payload.get("passed", False))
    package_passed = bool(package_skipped or package_code == 0)
    passed = bool(gov_passed and report_passed and package_passed)

    out = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "release_candidate_check_failed",
        "passed": passed,
        "freeze_dir": str(freeze_dir),
        "campaign_dir": str(campaign_dir),
        "manifest_path": str(manifest_path),
        "runs_path": str(runs_path),
        "stages": {
            "release_governance": {
                "passed": gov_passed,
                "returncode": gov_code,
                "payload": gov_payload,
            },
            "release_report": {
                "passed": report_passed,
                "returncode": report_code,
                "payload": report_payload,
            },
            "release_package": {
                "passed": package_passed,
                "returncode": package_code,
                "skipped": package_skipped,
                "package_out_dir": package_out_dir,
                "payload": package_payload,
            },
        },
    }
    if args.out:
        write_json(args.out, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if passed else 2
