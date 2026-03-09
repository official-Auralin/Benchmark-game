"""Pilot, freeze, release-package, and pilot-analysis commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

from ..baselines import make_agent
from ..checks import evaluate_suite
from ..generator import generate_suite
from ..io import (
    build_split_manifest,
    load_instance_bundle,
    load_json,
    load_jsonl,
    migrate_run_rows,
    run_record_to_dict,
    run_row_from_play_payload,
    validate_manifest,
    validate_run_rows,
    write_json,
    write_jsonl,
)
from ..meta import (
    ADAPTATION_POLICY_VERSION,
    ALLOWED_MODES,
    BASELINE_PANEL_CORE,
    BASELINE_PANEL_FULL,
    BASELINE_PANEL_POLICY_VERSION,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    COMPLEXITY_KNOB_KEYS,
    COMPLEXITY_POLICY_VERSION,
    COMPLEXITY_SCORE_METHOD,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    IDENTIFIABILITY_METRIC_ID,
    IDENTIFIABILITY_POLICY_VERSION,
    INSTANCE_BUNDLE_SCHEMA_VERSION,
    PILOT_FREEZE_SCHEMA_VERSION,
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
    baseline_panel_policy_message,
    build_report_payload,
    canonical_baseline_agent_id,
    panel_ids,
    renderer_policy_message,
    track_for_agent_id,
    track_tool_policy_message,
    validate_runs_manifest,
)


def parse_seed_list(seed_text: str) -> list[int]:
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


def run_freeze_pilot(
    *,
    freeze_id: str,
    split: str,
    seed_start: int,
    count: int,
    seeds_text: str,
    mode: str | None,
    out_dir_text: str,
    force: bool,
) -> tuple[int, dict[str, object]]:
    cfg = GeneratorConfig()
    git_commit = require_git_commit()
    mode_override = str(mode).strip() if mode is not None else ""
    if mode_override and mode_override not in ALLOWED_MODES:
        return 2, {
            "status": "error",
            "error_type": "unsupported_mode",
            "message": f"mode must be one of {list(ALLOWED_MODES)}",
            "mode": mode_override,
        }

    if seeds_text.strip():
        seeds = parse_seed_list(seeds_text)
    else:
        seeds = [int(seed_start) + i for i in range(int(count))]
    if not seeds:
        return 2, {
            "status": "error",
            "error_type": "empty_seed_set",
            "message": "pilot freeze requires at least one seed",
        }

    out_dir = Path(out_dir_text)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return 2, {
            "status": "error",
            "error_type": "output_dir_not_empty",
            "message": "output directory exists and is not empty; use --force to overwrite",
            "out_dir": str(out_dir),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "instance_bundle_v1.json"
    manifest_path = out_dir / "split_manifest_v1.json"
    freeze_path = out_dir / "pilot_freeze_v1.json"
    receipt_path = out_dir / "build_receipt.json"

    suite = generate_suite(
        seeds=seeds,
        split_id=split,
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
        "git_commit": git_commit,
        "config_hash": config_hash(cfg),
        "split_id": split,
        "seed_start": int(min(seeds)),
        "count": len(seeds),
        "seeds": seeds,
        "mode_override": mode_override or "mixed",
        "identifiability_policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "identifiability_metric_id": IDENTIFIABILITY_METRIC_ID,
        "identifiability_min_response_ratio": float(cfg.ident_min_response_ratio),
        "identifiability_min_unique_signatures": int(cfg.ident_min_unique_signatures),
        "instances_hash": stable_hash_json(instances_payload),
        "instances": instances_payload,
    }
    write_json(str(bundle_path), bundle)

    manifest = build_split_manifest(
        suite,
        bundle_meta=bundle,
    )
    manifest_errors = validate_manifest(manifest, strict=True)
    if manifest_errors:
        return 2, {
            "status": "error",
            "error_type": "manifest_validation_failed",
            "errors": manifest_errors,
        }
    write_json(str(manifest_path), manifest)

    freeze_meta = {
        "schema_version": PILOT_FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "provisional": True,
        "purpose": "internal_pilot",
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": git_commit,
        "config_hash": config_hash(cfg),
        "split_id": split,
        "seed_count": len(seeds),
        "seeds": seeds,
        "mode_override": mode_override or "mixed",
        "instance_bundle_hash": stable_hash_json(bundle),
        "split_manifest_hash": stable_hash_json(manifest),
        "identifiability_policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "identifiability_metric_id": IDENTIFIABILITY_METRIC_ID,
        "identifiability_min_response_ratio": float(cfg.ident_min_response_ratio),
        "identifiability_min_unique_signatures": int(cfg.ident_min_unique_signatures),
        "notes": (
            "Provisional pilot freeze for internal testing only. "
            "Does not finalize public/private split governance."
        ),
    }
    write_json(str(freeze_path), freeze_meta)
    write_json(
        str(receipt_path),
        {
            "status": "ok",
            "receipt_type": "build_receipt",
            "generated_on": date.today().isoformat(),
            "git_commit": git_commit,
            "output_root": str(out_dir),
            "instance_bundle_path": str(bundle_path),
            "split_manifest_path": str(manifest_path),
            "pilot_freeze_path": str(freeze_path),
        },
    )

    summary = {
        "status": "ok",
        "freeze_id": freeze_id,
        "out_dir": str(out_dir),
        "seed_count": len(seeds),
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
        "freeze_meta_path": str(freeze_path),
        "build_receipt_path": str(receipt_path),
        "instance_count": len(suite),
        "split_id": split,
        "mode_override": mode_override or "mixed",
        "identifiability_policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "identifiability_metric_id": IDENTIFIABILITY_METRIC_ID,
        "identifiability_min_response_ratio": float(cfg.ident_min_response_ratio),
        "identifiability_min_unique_signatures": int(cfg.ident_min_unique_signatures),
    }
    return 0, summary


def cmd_freeze_pilot(args: argparse.Namespace) -> int:
    code, payload = run_freeze_pilot(
        freeze_id=str(args.freeze_id),
        split=str(args.split),
        seed_start=int(args.seed_start),
        count=int(args.count),
        seeds_text=str(args.seeds),
        mode=args.mode,
        out_dir_text=str(args.out_dir),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def run_p0_seed_pack(
    *,
    freeze_id: str,
    split: str,
    seed_start: int,
    count: int,
    seeds_text: str,
    mode: str | None,
    out_dir_text: str,
    force: bool,
) -> tuple[int, dict[str, object]]:
    return run_freeze_pilot(
        freeze_id=freeze_id,
        split=split,
        seed_start=seed_start,
        count=count,
        seeds_text=seeds_text,
        mode=mode,
        out_dir_text=out_dir_text,
        force=force,
    )


def load_external_episode_payloads(path: str) -> list[dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"external episode path does not exist: {path}")

    if file_path.suffix.lower() == ".jsonl":
        payloads = load_jsonl(str(file_path))
    else:
        raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw_payload, dict):
            payloads = [raw_payload]
        elif isinstance(raw_payload, list):
            payloads = raw_payload
        else:
            raise ValueError(
                "external episode payload must be an object/list (JSON) or JSONL"
            )

    if not payloads:
        raise ValueError(f"external episode payload is empty: {path}")
    if any(not isinstance(item, dict) for item in payloads):
        raise ValueError("external episode payload rows must be JSON objects")
    return [{str(k): v for k, v in item.items()} for item in payloads]


def cmd_pilot_campaign(args: argparse.Namespace) -> int:
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

    renderer_track = str(args.renderer_track).strip()
    renderer_profile_id = renderer_profile_for_track(renderer_track)
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

    instances, bundle_meta = load_instance_bundle(str(bundle_path))
    git_commit = require_git_commit()
    instance_lookup = {inst.instance_id: inst for inst in instances}
    rows: list[dict[str, object]] = []
    panel_raw = panel_ids(args.baseline_panel)
    panel: list[str] = []
    seen_panel: set[str] = set()
    for item in panel_raw:
        try:
            canonical_id = canonical_baseline_agent_id(item)
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "baseline_panel_invalid",
                        "message": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if canonical_id not in seen_panel:
            panel.append(canonical_id)
            seen_panel.add(canonical_id)

    baseline_policy_level = str(args.baseline_policy_level).strip()
    baseline_policy_msg = baseline_panel_policy_message(
        panel,
        policy_level=baseline_policy_level,
    )
    if baseline_policy_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "baseline_panel_policy_violation",
                    "baseline_policy_version": BASELINE_PANEL_POLICY_VERSION,
                    "baseline_policy_level": baseline_policy_level,
                    "message": baseline_policy_msg,
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

    for idx, agent_id in enumerate(panel):
        agent = make_agent(agent_id)
        eval_track = track_for_agent_id(agent_id)
        if eval_track == "EVAL-TA":
            tool_allowlist_id = args.tool_allowlist_id
            tool_log_hash = args.tool_log_hash or stable_hash_json(
                {
                    "instances_hash": bundle_meta.get("instances_hash", ""),
                    "agent": agent.name,
                    "seed": int(args.seed + idx),
                    "panel_index": idx,
                }
            )[:16]
        elif eval_track == "EVAL-OC":
            tool_allowlist_id = DEFAULT_TOOL_ALLOWLIST_BY_TRACK["EVAL-OC"]
            tool_log_hash = stable_hash_json(
                {
                    "instances_hash": bundle_meta.get("instances_hash", ""),
                    "agent": agent.name,
                    "seed": int(args.seed + idx),
                    "panel_index": idx,
                    "policy": "oracle-ceiling",
                }
            )[:16]
        else:
            tool_allowlist_id = "none"
            tool_log_hash = ""

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
            renderer_track=renderer_track,
            seed=args.seed + idx,
        )
        run_meta = {
            "family_id": bundle_meta.get("family_id", FAMILY_ID),
            "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
            "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
            "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
            "harness_version": HARNESS_VERSION,
            "git_commit": git_commit,
            "config_hash": bundle_meta.get("config_hash", "unknown"),
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

    external_episode_rows_count = 0
    external_episode_run_meta = {
        "family_id": bundle_meta.get("family_id", FAMILY_ID),
        "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
        "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
        "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
        "harness_version": HARNESS_VERSION,
        "git_commit": git_commit,
        "config_hash": bundle_meta.get("config_hash", "unknown"),
    }
    for external_episode_path in args.external_episodes:
        try:
            payloads = load_external_episode_payloads(external_episode_path)
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "external_episode_invalid",
                        "path": external_episode_path,
                        "message": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        for payload_idx, payload in enumerate(payloads):
            try:
                row = run_row_from_play_payload(
                    payload,
                    run_meta=external_episode_run_meta,
                )
            except ValueError as exc:
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "error_type": "external_episode_invalid",
                            "path": external_episode_path,
                            "payload_index": payload_idx,
                            "message": str(exc),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 2
            rows.append(row)
            external_episode_rows_count += 1

    runs_path = out_dir / "runs_combined.jsonl"
    write_jsonl(str(runs_path), rows)

    err_payload, coverage_payload = validate_runs_manifest(
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

    validation_ok: dict[str, object] = {
        "status": "ok",
        "strict_mode": True,
        "official_mode": True,
        "rows": len(rows),
    }
    if coverage_payload is not None:
        validation_ok["manifest_coverage"] = coverage_payload
    write_json(str(validation_path), validation_ok)

    report = build_report_payload(
        rows=rows,
        strict_mode=True,
        coverage_payload=coverage_payload,
    )
    write_json(str(report_path), report)
    receipt_path = out_dir / "build_receipt.json"
    write_json(
        str(receipt_path),
        {
            "status": "ok",
            "receipt_type": "build_receipt",
            "generated_on": date.today().isoformat(),
            "git_commit": git_commit,
            "freeze_dir": str(freeze_dir),
            "out_dir": str(out_dir),
            "manifest_path": str(manifest_path),
            "runs_path": str(runs_path),
            "validation_path": str(validation_path),
            "report_path": str(report_path),
        },
    )

    summary = {
        "status": "ok",
        "freeze_dir": str(freeze_dir),
        "out_dir": str(out_dir),
        "runs_path": str(runs_path),
        "validation_path": str(validation_path),
        "report_path": str(report_path),
        "build_receipt_path": str(receipt_path),
        "row_count": len(rows),
        "renderer_track": renderer_track,
        "baseline_policy_version": BASELINE_PANEL_POLICY_VERSION,
        "baseline_policy_level": baseline_policy_level,
        "baseline_panel_required": (
            list(BASELINE_PANEL_FULL)
            if baseline_policy_level == "full"
            else list(BASELINE_PANEL_CORE)
        ),
        "renderer_policy_version": RENDERER_POLICY_VERSION,
        "renderer_profile_id": renderer_profile_id,
        "baseline_panel": panel,
        "external_runs_count": len(args.external_runs),
        "external_episode_paths_count": len(args.external_episodes),
        "external_episode_rows_count": int(external_episode_rows_count),
        "adaptation_condition": adaptation_condition,
        "adaptation_budget_tokens": adaptation_budget_tokens,
        "adaptation_data_scope": adaptation_data_scope,
        "adaptation_protocol_id": adaptation_protocol_id or "none",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cmd_release_package(args: argparse.Namespace) -> int:
    freeze_dir = Path(args.freeze_dir)
    campaign_dir = Path(args.campaign_dir)
    out_dir = Path(args.out_dir)

    required_sources = {
        "instance_bundle_v1.json": freeze_dir / "instance_bundle_v1.json",
        "split_manifest_v1.json": freeze_dir / "split_manifest_v1.json",
        "runs_combined.jsonl": campaign_dir / "runs_combined.jsonl",
        "official_validation.json": campaign_dir / "official_validation.json",
        "official_report.json": campaign_dir / "official_report.json",
    }
    missing_sources = [
        {"name": name, "path": str(path)}
        for name, path in required_sources.items()
        if not path.exists()
    ]
    if missing_sources:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "release_package_source_missing",
                    "missing": missing_sources,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

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

    rows = load_jsonl(str(required_sources["runs_combined.jsonl"]))
    err_payload, coverage_payload = validate_runs_manifest(
        rows,
        manifest_path=str(required_sources["split_manifest_v1.json"]),
        strict_mode=True,
        official_mode=True,
    )
    if err_payload is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "release_package_source_invalid",
                    "validation_error": err_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        required_sources["instance_bundle_v1.json"],
        required_sources["split_manifest_v1.json"],
        freeze_dir / "pilot_freeze_v1.json",
        required_sources["runs_combined.jsonl"],
        required_sources["official_validation.json"],
        required_sources["official_report.json"],
        campaign_dir / "pilot_analysis.json",
    ]
    copied_files: list[Path] = []
    for src in files_to_copy:
        if not src.exists():
            continue
        dst = artifacts_dir / src.name
        shutil.copy2(src, dst)
        copied_files.append(dst)

    instructions_path = out_dir / "RERUN_INSTRUCTIONS.md"
    instructions_lines = [
        "# GF-01 Reproducibility Package Instructions",
        "",
        "## 1) Validate strict/official run schema and manifest coverage",
        "```bash",
        "python3 -m gf01 validate \\",
        "  --runs artifacts/runs_combined.jsonl \\",
        "  --manifest artifacts/split_manifest_v1.json \\",
        "  --official",
        "```",
        "",
        "## 2) Recompute grouped report from packaged runs",
        "```bash",
        "python3 -m gf01 report \\",
        "  --runs artifacts/runs_combined.jsonl \\",
        "  --manifest artifacts/split_manifest_v1.json \\",
        "  --official \\",
        "  --out artifacts/recomputed_report.json",
        "```",
        "",
        "## 3) Run benchmark integrity checks",
        "```bash",
        "python3 -m gf01 checks --seed 3000",
        "```",
        "",
        "## Notes",
        "- Compare `artifacts/recomputed_report.json` against packaged",
        "  `artifacts/official_report.json`.",
        "- The packaged `split_manifest_v1.json` is the official coverage target.",
    ]
    instructions_path.write_text("\n".join(instructions_lines) + "\n", encoding="utf-8")

    file_entries = []
    for artifact in sorted(copied_files, key=lambda path: path.name):
        file_entries.append(
            {
                "path": f"artifacts/{artifact.name}",
                "size_bytes": int(artifact.stat().st_size),
                "sha256": sha256_file(artifact),
            }
        )
    file_entries.append(
        {
            "path": instructions_path.name,
            "size_bytes": int(instructions_path.stat().st_size),
            "sha256": sha256_file(instructions_path),
        }
    )

    package_manifest = {
        "schema_version": "gf01.release_package.v1",
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": require_git_commit(),
        "strict_validation": {
            "status": "ok",
            "rows": len(rows),
            "manifest_coverage": coverage_payload or {},
        },
        "files": file_entries,
    }
    manifest_path = out_dir / "release_package_manifest.json"
    write_json(str(manifest_path), package_manifest)
    receipt_path = out_dir / "build_receipt.json"
    write_json(
        str(receipt_path),
        {
            "status": "ok",
            "receipt_type": "build_receipt",
            "created_on": date.today().isoformat(),
            "git_commit": package_manifest["git_commit"],
            "freeze_dir": str(freeze_dir),
            "campaign_dir": str(campaign_dir),
            "out_dir": str(out_dir),
        },
    )

    summary = {
        "status": "ok",
        "schema_version": "gf01.release_package.v1",
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "instructions_path": str(instructions_path),
        "build_receipt_path": str(receipt_path),
        "artifact_count": len(copied_files),
        "source_freeze_dir": str(freeze_dir),
        "source_campaign_dir": str(campaign_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def analysis_rate(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(field, False))) / float(len(rows))


def complexity_values(row: dict[str, object]) -> dict[str, float]:
    complexity = row.get("complexity")
    if not isinstance(complexity, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in complexity.items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def complexity_score(row: dict[str, object]) -> float | None:
    complexity = complexity_values(row)
    if not complexity:
        return None
    return statistics.fmean(complexity.values())


def assign_numeric_quartiles(
    rows: list[dict[str, object]],
    score_getter,
    *,
    score_field: str,
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for row in rows:
        score = score_getter(row)
        if score is None:
            continue
        cloned = dict(row)
        cloned[score_field] = float(score)
        scored.append(cloned)
    scored.sort(key=lambda item: float(item[score_field]))
    n = len(scored)
    if n == 0:
        return []
    for idx, row in enumerate(scored):
        quartile_idx = min(4, (idx * 4) // n + 1)
        row["quartile"] = f"Q{quartile_idx}"
    return scored


def assign_complexity_quartiles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return assign_numeric_quartiles(
        rows,
        complexity_score,
        score_field="complexity_score",
    )


def quartile_stats(
    rows: list[dict[str, object]],
    *,
    score_field: str,
) -> list[dict[str, object]]:
    quartile_rows: dict[str, list[dict[str, object]]] = {q: [] for q in ("Q1", "Q2", "Q3", "Q4")}
    for row in rows:
        quartile = str(row.get("quartile", "Q4"))
        quartile_rows.setdefault(quartile, []).append(row)
    out: list[dict[str, object]] = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        bucket = quartile_rows.get(quartile, [])
        stats_row: dict[str, object] = {
            "quartile": quartile,
            "count": len(bucket),
            "certified_rate": analysis_rate(bucket, "valid"),
            "goal_rate": analysis_rate(bucket, "goal"),
        }
        if bucket:
            values = [float(item.get(score_field, 0.0)) for item in bucket]
            stats_row[f"{score_field}_min"] = min(values)
            stats_row[f"{score_field}_max"] = max(values)
        out.append(stats_row)
    return out


def pearson_corr(x_vals: list[float], y_vals: list[float]) -> float | None:
    if len(x_vals) < 2 or len(y_vals) < 2 or len(x_vals) != len(y_vals):
        return None
    x_mean = statistics.fmean(x_vals)
    y_mean = statistics.fmean(y_vals)
    x_var = statistics.fmean((x - x_mean) ** 2 for x in x_vals)
    y_var = statistics.fmean((y - y_mean) ** 2 for y in y_vals)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = statistics.fmean((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    return cov / math.sqrt(x_var * y_var)


def complexity_knob_diagnostics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for knob in COMPLEXITY_KNOB_KEYS:
        knob_rows: list[dict[str, object]] = []
        knob_values: list[float] = []
        valid_values: list[float] = []
        goal_values: list[float] = []
        for row in rows:
            values = complexity_values(row)
            if knob not in values:
                continue
            knob_value = float(values[knob])
            cloned = dict(row)
            cloned["knob_value"] = knob_value
            knob_rows.append(cloned)
            knob_values.append(knob_value)
            valid_values.append(1.0 if bool(row.get("valid", False)) else 0.0)
            goal_values.append(1.0 if bool(row.get("goal", False)) else 0.0)
        value_min = min(knob_values) if knob_values else None
        value_max = max(knob_values) if knob_values else None
        is_constant = (
            value_min is not None
            and value_max is not None
            and abs(float(value_max) - float(value_min)) <= 1e-12
        )
        q1_certified_rate: float | None = None
        q4_certified_rate: float | None = None
        q1_minus_q4_certified_rate: float | None = None
        q1_goal_rate: float | None = None
        q4_goal_rate: float | None = None
        q1_minus_q4_goal_rate: float | None = None
        if is_constant:
            quartiles = [
                {
                    "quartile": "ALL",
                    "count": len(knob_rows),
                    "certified_rate": analysis_rate(knob_rows, "valid"),
                    "goal_rate": analysis_rate(knob_rows, "goal"),
                    "knob_value_min": value_min,
                    "knob_value_max": value_max,
                }
            ]
        else:
            quartiled = assign_numeric_quartiles(
                knob_rows,
                lambda row: row.get("knob_value"),
                score_field="knob_value",
            )
            quartiles = quartile_stats(quartiled, score_field="knob_value")
            q1 = quartiles[0]
            q4 = quartiles[3]
            q1_certified_rate = float(q1["certified_rate"])
            q4_certified_rate = float(q4["certified_rate"])
            q1_minus_q4_certified_rate = q1_certified_rate - q4_certified_rate
            q1_goal_rate = float(q1["goal_rate"])
            q4_goal_rate = float(q4["goal_rate"])
            q1_minus_q4_goal_rate = q1_goal_rate - q4_goal_rate
        diagnostics.append(
            {
                "knob": knob,
                "row_count": len(knob_rows),
                "is_constant": is_constant,
                "value_min": value_min,
                "value_max": value_max,
                "q1_certified_rate": q1_certified_rate,
                "q4_certified_rate": q4_certified_rate,
                "q1_minus_q4_certified_rate": q1_minus_q4_certified_rate,
                "q1_goal_rate": q1_goal_rate,
                "q4_goal_rate": q4_goal_rate,
                "q1_minus_q4_goal_rate": q1_minus_q4_goal_rate,
                "pearson_corr_certified": pearson_corr(knob_values, valid_values),
                "pearson_corr_goal": pearson_corr(knob_values, goal_values),
                "quartile_stats": quartiles,
            }
        )
    diagnostics.sort(
        key=lambda row: (
            -abs(float(row.get("q1_minus_q4_certified_rate", 0.0) or 0.0)),
            str(row.get("knob", "")),
        )
    )
    return diagnostics


def agent_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str, str, str, str, bool, str, int, str, str],
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
            str(row.get("renderer_track", "unknown")),
            str(row.get("renderer_profile_id", "unknown")),
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
            renderer_track,
            renderer_profile_id,
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
                "renderer_track": renderer_track,
                "renderer_profile_id": renderer_profile_id,
                "play_protocol": play_protocol,
                "scored_commit_episode": scored_commit_episode,
                "adaptation_condition": adaptation_condition,
                "adaptation_budget_tokens": adaptation_budget_tokens,
                "adaptation_data_scope": adaptation_data_scope,
                "adaptation_protocol_id": adaptation_protocol_id,
                "count": len(entries),
                "goal_rate": analysis_rate(entries, "goal"),
                "certified_rate": analysis_rate(entries, "valid"),
                "ap_f1_mean": statistics.fmean(float(e.get("ap_f1", 0.0)) for e in entries),
                "ts_f1_mean": statistics.fmean(float(e.get("ts_f1", 0.0)) for e in entries),
            }
        )
    return summary_rows


def cmd_pilot_analyze(args: argparse.Namespace) -> int:
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
    migration_payload: dict[str, object] | None = None
    if schema_errors:
        migrated_rows, migration_report = migrate_run_rows(
            rows,
            defaults={
                "schema_version": RUN_RECORD_SCHEMA_VERSION,
                "family_id": FAMILY_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "generator_version": GENERATOR_VERSION,
                "checker_version": CHECKER_VERSION,
                "harness_version": HARNESS_VERSION,
                "git_commit": require_git_commit(),
                "config_hash": "pilot-analyze-legacy-backfill",
                "tool_allowlist_id": DEFAULT_TOOL_ALLOWLIST_BY_TRACK["EVAL-CB"],
                "tool_log_hash": "",
                "play_protocol": "commit_only",
                "scored_commit_episode": True,
                "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
                "adaptation_condition": "no_adaptation",
                "adaptation_budget_tokens": 0,
                "adaptation_data_scope": "none",
                "adaptation_protocol_id": "none",
                "eval_track": "EVAL-CB",
                "renderer_track": "json",
                "renderer_policy_version": RENDERER_POLICY_VERSION,
                "renderer_profile_id": renderer_profile_for_track("json"),
                "agent_name": "legacy-agent",
                "split_id": "public_dev",
                "mode": "normal",
                "seed": 0,
            },
        )
        migrated_errors = validate_run_rows(migrated_rows, strict=True)
        if not migrated_errors:
            rows = migrated_rows
            migration_payload = migration_report
            schema_errors = []
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
    public_splits = {
        token.strip() for token in str(args.public_splits).split(",") if token.strip()
    }

    mode_rows = [
        row
        for row in rows
        if str(row.get("eval_track", "")) == eval_track
        and str(row.get("mode", "")) == mode
        and bool(row.get("scored_commit_episode", True))
    ]
    greedy_rows = [
        row for row in mode_rows if str(row.get("agent_name", "")) == greedy_agent_name
    ]
    held_out_greedy_rows = [
        row for row in greedy_rows if str(row.get("split_id", "")) not in public_splits
    ]
    if not held_out_greedy_rows:
        held_out_greedy_rows = greedy_rows

    quartiled = assign_complexity_quartiles(held_out_greedy_rows)
    held_out_quartiles = quartile_stats(quartiled, score_field="complexity_score")

    q1_stats = held_out_quartiles[0]
    q4_stats = held_out_quartiles[3]
    m_q1 = float(q1_stats["certified_rate"])
    m_q4 = float(q4_stats["certified_rate"])
    discrimination_delta = m_q1 - m_q4
    discrimination_trigger = (
        discrimination_delta < float(args.discrimination_delta_threshold)
        or m_q4 < float(args.discrimination_q4_floor)
    )

    greedy_goal = analysis_rate(held_out_greedy_rows, "goal")
    greedy_m = analysis_rate(held_out_greedy_rows, "valid")
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
    oracle_minus_greedy = analysis_rate(oracle_rows, "valid") - greedy_m if oracle_rows else None

    recommendation = "keep_coefficients"
    if discrimination_trigger or shortcut_trigger:
        recommendation = "recalibrate_normal_window"

    mode_rows_same_track = [
        row
        for row in mode_rows
        if str(row.get("eval_track", "")) == eval_track
        and bool(row.get("scored_commit_episode", True))
    ]
    pooled_quartiled = assign_complexity_quartiles(mode_rows_same_track)
    pooled_quartiles = quartile_stats(
        pooled_quartiled,
        score_field="complexity_score",
    )
    pooled_q1 = pooled_quartiles[0]
    pooled_q4 = pooled_quartiles[3]
    pooled_knob_stats = complexity_knob_diagnostics(mode_rows_same_track)
    held_out_knob_stats = complexity_knob_diagnostics(held_out_greedy_rows)
    per_agent_knob_stats = {
        agent_name: complexity_knob_diagnostics(
            [
                row
                for row in mode_rows_same_track
                if str(row.get("agent_name", "")) == agent_name
            ]
        )
        for agent_name in sorted(
            {str(row.get("agent_name", "")) for row in mode_rows_same_track}
        )
        if agent_name
    }

    out = {
        "status": "ok",
        "policy_reference": "DEC-014d",
        "complexity_policy_version": COMPLEXITY_POLICY_VERSION,
        "complexity_policy": {
            "score_method": COMPLEXITY_SCORE_METHOD,
            "knob_keys": list(COMPLEXITY_KNOB_KEYS),
        },
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
            "quartile_stats": held_out_quartiles,
            "triggered": discrimination_trigger,
        },
        "shortcut_check": {
            "goal_rate": greedy_goal,
            "certified_rate": greedy_m,
            "triggered": shortcut_trigger,
            "row_count": len(held_out_greedy_rows),
        },
        "complexity_diagnostics": {
            "scope_rows": {
                "held_out_greedy": len(held_out_greedy_rows),
                "pooled_eval_track_mode": len(mode_rows_same_track),
            },
            "held_out_greedy_composite": {
                "q1_certified_rate": float(q1_stats["certified_rate"]),
                "q4_certified_rate": float(q4_stats["certified_rate"]),
                "q1_minus_q4_certified_rate": discrimination_delta,
                "quartile_stats": held_out_quartiles,
            },
            "pooled_eval_track_mode_composite": {
                "q1_certified_rate": float(pooled_q1["certified_rate"]),
                "q4_certified_rate": float(pooled_q4["certified_rate"]),
                "q1_minus_q4_certified_rate": float(pooled_q1["certified_rate"])
                - float(pooled_q4["certified_rate"]),
                "quartile_stats": pooled_quartiles,
            },
            "held_out_greedy_knob_stats": held_out_knob_stats,
            "pooled_eval_track_mode_knob_stats": pooled_knob_stats,
            "per_agent_knob_stats": per_agent_knob_stats,
        },
        "oracle_context": {
            "oracle_minus_greedy_certified_rate": oracle_minus_greedy,
            "oracle_row_count": len(oracle_rows),
        },
        "agent_summary": agent_summary_rows(rows),
        "recommendation": recommendation,
    }
    if validation_payload is not None:
        out["official_validation"] = validation_payload
    if migration_payload is not None:
        out["legacy_migration"] = {
            "applied": True,
            "report": migration_payload,
        }

    out_path = Path(args.out) if args.out else campaign_dir / "pilot_analysis.json"
    write_json(str(out_path), out)
    out["out_path"] = str(out_path)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0
