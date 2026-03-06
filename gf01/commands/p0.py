"""P0 setup and gate command implementations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..io import write_json
from .pilot import run_p0_seed_pack


def load_feedback_rows(
    feedback_path: Path,
    *,
    required_columns: set[str],
) -> tuple[list[dict[str, str]] | None, dict[str, object] | None]:
    if not feedback_path.exists():
        return None, {
            "status": "error",
            "error_type": "missing_feedback_file",
            "message": f"feedback CSV not found: {feedback_path}",
        }

    with feedback_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required_columns.difference(columns))
        if missing:
            return None, {
                "status": "error",
                "error_type": "feedback_schema_error",
                "message": (
                    "feedback CSV missing required columns: " + ", ".join(missing)
                ),
            }
        rows = [dict(row) for row in reader]

    if not rows:
        return None, {
            "status": "error",
            "error_type": "empty_feedback",
            "message": "feedback CSV contains no rows",
        }
    return rows, None


def run_p0_feedback_check(
    *,
    feedback_path: Path,
    min_score: int,
    min_ratio: float,
) -> tuple[int, dict[str, object]]:
    rows, error = load_feedback_rows(
        feedback_path,
        required_columns={
            "tester_id",
            "objective_clarity",
            "control_clarity",
            "action_effect_clarity",
            "must_fix_blockers",
        },
    )
    if error is not None:
        return 2, error
    assert rows is not None

    bad_rows: list[str] = []
    objective_ok = 0
    control_ok = 0
    action_effect_ok = 0
    must_fix_total = 0
    tester_ids: list[str] = []
    for idx, row in enumerate(rows, start=1):
        tester_id = str(row.get("tester_id", "")).strip() or f"row-{idx}"
        tester_ids.append(tester_id)
        try:
            objective = int(str(row.get("objective_clarity", "")).strip())
            control = int(str(row.get("control_clarity", "")).strip())
            action_effect = int(str(row.get("action_effect_clarity", "")).strip())
            must_fix = int(str(row.get("must_fix_blockers", "")).strip())
        except ValueError:
            bad_rows.append(tester_id)
            continue
        if objective >= min_score:
            objective_ok += 1
        if control >= min_score:
            control_ok += 1
        if action_effect >= min_score:
            action_effect_ok += 1
        must_fix_total += max(0, must_fix)

    if bad_rows:
        return 2, {
            "status": "error",
            "error_type": "feedback_value_error",
            "message": "non-integer clarity/blocker fields in rows",
            "bad_rows": bad_rows,
        }

    n = len(rows)
    objective_ratio = objective_ok / n
    control_ratio = control_ok / n
    action_effect_ratio = action_effect_ok / n
    passed = (
        must_fix_total == 0
        and objective_ratio >= min_ratio
        and control_ratio >= min_ratio
        and action_effect_ratio >= min_ratio
    )
    payload = {
        "status": "ok" if passed else "error",
        "schema_version": "gf01.p0_feedback_check.v1",
        "feedback_path": str(feedback_path),
        "tester_count": n,
        "testers": tester_ids,
        "thresholds": {
            "min_score": min_score,
            "min_ratio": min_ratio,
            "must_fix_total_required": 0,
        },
        "metrics": {
            "objective_ratio_ge_min_score": objective_ratio,
            "control_ratio_ge_min_score": control_ratio,
            "action_effect_ratio_ge_min_score": action_effect_ratio,
            "must_fix_total": must_fix_total,
        },
        "passed": passed,
    }
    if not passed:
        payload["error_type"] = "feedback_thresholds_not_met"
    return (0 if passed else 2), payload


def cmd_p0_feedback_check(args: argparse.Namespace) -> int:
    code, payload = run_p0_feedback_check(
        feedback_path=Path(args.feedback),
        min_score=int(args.min_score),
        min_ratio=float(args.min_ratio),
    )
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def run_p0_feedback_template(
    *,
    out_path: Path,
    force: bool,
) -> tuple[int, dict[str, object]]:
    if out_path.exists() and not force:
        return 2, {
            "status": "error",
            "error_type": "template_exists",
            "message": (
                f"template output already exists: {out_path}; "
                "use --force to overwrite"
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "tester_id,date,backend_used,seed_list_run,objective_clarity,control_clarity,action_effect_clarity,visual_overload,must_fix_blockers,notes",
        "tester_01,YYYY-MM-DD,pygame,7000;7001;7002,3,3,3,2,0,",
        "tester_02,YYYY-MM-DD,text,7000;7001;7002,3,3,3,2,0,",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0, {
        "status": "ok",
        "template_schema_version": "gf01.p0_feedback_template.v1",
        "out": str(out_path),
        "rows_written": 2,
    }


def cmd_p0_feedback_template(args: argparse.Namespace) -> int:
    code, payload = run_p0_feedback_template(
        out_path=Path(args.out),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def parse_seed_list_flexible(seed_text: str) -> list[int]:
    normalized = seed_text.replace(";", ",")
    seeds: list[int] = []
    for chunk in normalized.split(","):
        for token in chunk.strip().split():
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


def run_p0_session_check(
    *,
    feedback_path: Path,
    runs_dir: Path,
    required_renderer_track: str,
) -> tuple[int, dict[str, object]]:
    if not runs_dir.exists():
        return 2, {
            "status": "error",
            "error_type": "missing_runs_dir",
            "message": f"runs directory not found: {runs_dir}",
        }

    rows, error = load_feedback_rows(
        feedback_path,
        required_columns={"tester_id", "backend_used", "seed_list_run"},
    )
    if error is not None:
        return 2, error
    assert rows is not None

    bad_seed_rows: list[str] = []
    expected_sessions = 0
    found_sessions = 0
    missing_artifacts: list[dict[str, object]] = []
    payload_issues: list[dict[str, object]] = []

    for idx, row in enumerate(rows, start=1):
        tester_id = str(row.get("tester_id", "")).strip() or f"row-{idx}"
        backend_used = str(row.get("backend_used", "")).strip().lower()
        seed_list_raw = str(row.get("seed_list_run", "")).strip()
        try:
            seeds = parse_seed_list_flexible(seed_list_raw)
        except ValueError:
            bad_seed_rows.append(tester_id)
            continue
        if not seeds:
            bad_seed_rows.append(tester_id)
            continue
        for seed in seeds:
            expected_sessions += 1
            artifact_path = runs_dir / f"{tester_id}_{seed}.json"
            if not artifact_path.exists():
                missing_artifacts.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                    }
                )
                continue
            found_sessions += 1
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "invalid_json",
                    }
                )
                continue
            if not isinstance(payload, dict):
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "payload_not_object",
                    }
                )
                continue

            if str(payload.get("status", "")).strip() != "ok":
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "payload_status_not_ok",
                    }
                )
                continue

            instance = payload.get("instance")
            if not isinstance(instance, dict):
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "missing_instance_object",
                    }
                )
                continue
            try:
                payload_seed = int(instance.get("seed"))
            except (TypeError, ValueError):
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "invalid_instance_seed",
                    }
                )
                continue
            if payload_seed != seed:
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "instance_seed_mismatch",
                        "payload_seed": payload_seed,
                    }
                )
                continue

            run_contract = payload.get("run_contract")
            if not isinstance(run_contract, dict):
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "missing_run_contract_object",
                    }
                )
                continue
            payload_renderer_track = str(run_contract.get("renderer_track", "")).strip()
            if payload_renderer_track != required_renderer_track:
                payload_issues.append(
                    {
                        "tester_id": tester_id,
                        "seed": seed,
                        "path": str(artifact_path),
                        "issue": "renderer_track_mismatch",
                        "required_renderer_track": required_renderer_track,
                        "payload_renderer_track": payload_renderer_track,
                    }
                )
                continue
            if backend_used:
                payload_backend = str(run_contract.get("visual_backend", "")).strip().lower()
                if payload_backend != backend_used:
                    payload_issues.append(
                        {
                            "tester_id": tester_id,
                            "seed": seed,
                            "path": str(artifact_path),
                            "issue": "visual_backend_mismatch",
                            "feedback_backend": backend_used,
                            "payload_backend": payload_backend,
                        }
                    )

    if bad_seed_rows:
        return 2, {
            "status": "error",
            "error_type": "feedback_seed_list_error",
            "message": "feedback rows contain invalid or empty seed_list_run values",
            "bad_rows": bad_seed_rows,
        }

    passed = not missing_artifacts and not payload_issues
    payload = {
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "p0_session_coverage_failed",
        "schema_version": "gf01.p0_session_check.v1",
        "feedback_path": str(feedback_path),
        "runs_dir": str(runs_dir),
        "required_renderer_track": required_renderer_track,
        "tester_count": len(rows),
        "expected_session_count": expected_sessions,
        "found_session_count": found_sessions,
        "missing_session_count": len(missing_artifacts),
        "payload_issue_count": len(payload_issues),
        "missing_sessions_preview": missing_artifacts[:50],
        "payload_issues_preview": payload_issues[:50],
        "passed": passed,
    }
    return (0 if passed else 2), payload


def cmd_p0_session_check(args: argparse.Namespace) -> int:
    code, payload = run_p0_session_check(
        feedback_path=Path(args.feedback),
        runs_dir=Path(args.runs_dir),
        required_renderer_track=str(args.required_renderer_track).strip(),
    )
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def run_p0_gate(
    *,
    feedback_path: Path,
    runs_dir: Path,
    required_renderer_track: str,
    min_score: int,
    min_ratio: float,
) -> tuple[int, dict[str, object]]:
    session_code, session_payload = run_p0_session_check(
        feedback_path=feedback_path,
        runs_dir=runs_dir,
        required_renderer_track=required_renderer_track,
    )
    if session_code != 0:
        return 2, {
            "status": "error",
            "error_type": "p0_gate_session_check_failed",
            "schema_version": "gf01.p0_gate.v1",
            "passed": False,
            "session_check": session_payload,
        }

    feedback_code, feedback_payload = run_p0_feedback_check(
        feedback_path=feedback_path,
        min_score=min_score,
        min_ratio=min_ratio,
    )
    if feedback_code != 0:
        return 2, {
            "status": "error",
            "error_type": "p0_gate_feedback_check_failed",
            "schema_version": "gf01.p0_gate.v1",
            "passed": False,
            "session_check": session_payload,
            "feedback_check": feedback_payload,
        }

    return 0, {
        "status": "ok",
        "schema_version": "gf01.p0_gate.v1",
        "passed": True,
        "session_check": session_payload,
        "feedback_check": feedback_payload,
    }


def cmd_p0_gate(args: argparse.Namespace) -> int:
    code, payload = run_p0_gate(
        feedback_path=Path(args.feedback),
        runs_dir=Path(args.runs_dir),
        required_renderer_track=str(args.required_renderer_track),
        min_score=int(args.min_score),
        min_ratio=float(args.min_ratio),
    )
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def run_p0_init(
    *,
    template_out: Path,
    freeze_id: str,
    split: str,
    seed_start: int,
    count: int,
    seeds_text: str,
    mode: str | None,
    out_dir_text: str,
    force: bool,
) -> tuple[int, dict[str, object]]:
    template_code, template_payload = run_p0_feedback_template(
        out_path=template_out,
        force=force,
    )
    if template_code != 0:
        return 2, {
            "status": "error",
            "error_type": "p0_init_template_failed",
            "schema_version": "gf01.p0_init.v1",
            "passed": False,
            "template": template_payload,
        }

    seed_pack_code, seed_pack_payload = run_p0_seed_pack(
        freeze_id=freeze_id,
        split=split,
        seed_start=seed_start,
        count=count,
        seeds_text=seeds_text,
        mode=mode,
        out_dir_text=out_dir_text,
        force=force,
    )
    if seed_pack_code != 0:
        return 2, {
            "status": "error",
            "error_type": "p0_init_seed_pack_failed",
            "schema_version": "gf01.p0_init.v1",
            "passed": False,
            "template": template_payload,
            "seed_pack": seed_pack_payload,
        }

    return 0, {
        "status": "ok",
        "schema_version": "gf01.p0_init.v1",
        "passed": True,
        "template": template_payload,
        "seed_pack": seed_pack_payload,
    }


def cmd_p0_init(args: argparse.Namespace) -> int:
    code, payload = run_p0_init(
        template_out=Path(args.template_out),
        freeze_id=str(args.freeze_id),
        split=str(args.split),
        seed_start=int(args.seed_start),
        count=int(args.count),
        seeds_text=str(args.seeds),
        mode=args.mode,
        out_dir_text=str(args.out_dir),
        force=bool(args.force),
    )
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(code)


def cmd_p0_seed_pack(args: argparse.Namespace) -> int:
    code, payload = run_p0_seed_pack(
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
