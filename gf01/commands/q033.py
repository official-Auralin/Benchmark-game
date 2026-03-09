"""Q033 manifest, sweep, and closure-check commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..meta import BASELINE_PANEL_FULL
from ..models import GeneratorConfig
from ..profiling import PerformanceGates
from ..q033 import (
    Q033_PROTOCOL_VERSION,
    build_q033_manifests,
    q033_closure_check,
    run_q033_sweep,
)
from ..io import load_json, write_json


def cmd_q033_build_manifests(args: argparse.Namespace) -> int:
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
    cfg = GeneratorConfig()
    try:
        payload = build_q033_manifests(
            seed_start=int(args.seed_start),
            candidate_count=int(args.candidate_count),
            replicates=int(args.replicates),
            per_quartile=int(args.per_quartile),
            split_id=str(args.split),
            cfg=cfg,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "q033_manifest_build_failed",
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    manifest_paths: list[str] = []
    for manifest in payload.get("manifests", []):
        manifest_id = str(manifest.get("manifest_id", "q033_manifest"))
        path = out_dir / f"{manifest_id}.json"
        write_json(str(path), manifest)
        manifest_paths.append(path.name)

    index_payload = {
        "status": "ok",
        "schema_version": str(payload.get("schema_version", "")),
        "protocol_version": str(payload.get("protocol_version", Q033_PROTOCOL_VERSION)),
        "manifest_count": len(manifest_paths),
        "manifest_paths": manifest_paths,
        "seed_start": int(payload.get("seed_start", int(args.seed_start))),
        "candidate_count": int(payload.get("candidate_count", int(args.candidate_count))),
        "replicate_count": int(payload.get("replicate_count", int(args.replicates))),
        "per_quartile": int(payload.get("per_quartile", int(args.per_quartile))),
        "required_per_quartile": int(
            payload.get(
                "required_per_quartile",
                int(args.per_quartile) * int(args.replicates),
            )
        ),
        "available_counts": payload.get("available_counts", {}),
        "manifest_seed_overlaps": payload.get("manifest_seed_overlaps", []),
    }
    index_path = out_dir / "q033_manifest_index.json"
    write_json(str(index_path), index_payload)
    receipt_path = out_dir / "build_receipt.json"
    write_json(
        str(receipt_path),
        {
            "status": "ok",
            "receipt_type": "build_receipt",
            "out_dir": str(out_dir),
            "index_path": str(index_path),
            "manifest_paths": [str(out_dir / path) for path in manifest_paths],
        },
    )
    summary_payload = dict(index_payload)
    summary_payload["build_receipt_path"] = str(receipt_path)
    summary_payload["index_path"] = str(index_path)
    summary_payload["out_dir"] = str(out_dir)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


def cmd_q033_sweep(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    panel: list[str] = []
    seen_panel: set[str] = set()
    for part in str(args.baseline_panel).split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in seen_panel:
            continue
        panel.append(token)
        seen_panel.add(token)
    if not panel:
        panel = list(BASELINE_PANEL_FULL)
    unknown = sorted(set(panel) - set(BASELINE_PANEL_FULL))
    if unknown:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "unsupported_baseline_panel",
                    "message": f"unsupported baseline ids: {unknown}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if "greedy" not in panel or "oracle" not in panel:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "q033_panel_missing_required_agents",
                    "message": "Q-033 sweep requires panel to include greedy and oracle",
                    "panel": panel,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    cfg = GeneratorConfig()
    gates = PerformanceGates(
        max_generate_ms_mean=float(args.max_generate_ms_mean),
        max_minset_ms_mean=float(args.max_minset_ms_mean),
        max_eval_ms_mean=float(args.max_eval_ms_mean),
        max_checks_total_ms=float(args.max_checks_total_ms),
        max_truncation_rate=float(args.max_truncation_rate),
        min_oracle_minus_greedy_certified_gap=float(args.min_oracle_minus_greedy_gap),
    )
    try:
        report = run_q033_sweep(
            manifest=manifest,
            panel=panel,
            cfg=cfg,
            gates=gates,
            seed_base=int(args.seed),
            max_quartile_truncation_rate=float(args.max_quartile_truncation_rate),
            min_quartile_oracle_minus_greedy_gap=float(args.min_quartile_gap),
            max_quartile_runtime_gate_failures=int(args.max_quartile_runtime_gate_failures),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "q033_sweep_failed",
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
    return 0 if bool(report.get("gates", {}).get("passed_all", False)) else 2


def cmd_q033_closure_check(args: argparse.Namespace) -> int:
    sweep_paths = list(args.sweep or [])
    if len(sweep_paths) < 2:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "insufficient_sweeps",
                    "message": "provide at least two --sweep inputs",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    sweeps: list[dict[str, object]] = []
    for path in sweep_paths:
        sweeps.append(load_json(path))
    try:
        report = q033_closure_check(
            sweeps=sweeps,
            require_disjoint_seeds=not bool(args.allow_seed_overlap),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "q033_closure_check_failed",
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
    return 0 if bool(report.get("close_q033", False)) else 2
