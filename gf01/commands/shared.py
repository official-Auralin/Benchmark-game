"""Shared helpers for grouped GF-01 CLI command modules."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from ..io import load_json, validate_manifest, validate_run_rows
from ..meta import (
    ALLOWED_ADAPTATION_CONDITIONS,
    ALLOWED_ADAPTATION_DATA_SCOPES,
    ALLOWED_BASELINE_PANEL_LEVELS,
    ALLOWED_RENDERER_TRACKS,
    ALLOWED_TOOL_ALLOWLISTS_BY_TRACK,
    BASELINE_PANEL_CORE,
    BASELINE_PANEL_FULL,
    OFFICIAL_SPLITS,
    ROTATION_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
    renderer_profile_for_track,
)


def compute_manifest_coverage(
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


def validate_runs_manifest(
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
        coverage, missing_ids, unexpected_ids = compute_manifest_coverage(rows, manifest)
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


def track_tool_policy_message(
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


def adaptation_policy_message(
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


def canonical_baseline_agent_id(agent_id: str) -> str:
    key = agent_id.strip().lower()
    if key in {"random", "bl-00", "bl-00-randomintervention"}:
        return "random"
    if key in {"greedy", "bl-01", "bl-01-greedylocal"}:
        return "greedy"
    if key in {"search", "bl-02", "bl-02-budgetawaresearch"}:
        return "search"
    if key in {"tool", "bl-03", "bl-03-toolplanner"}:
        return "tool"
    if key in {"oracle", "bl-04", "bl-04-exactoracle"}:
        return "oracle"
    raise ValueError(f"unknown baseline agent id: {agent_id}")


def baseline_panel_policy_message(
    panel_ids: list[str],
    *,
    policy_level: str,
) -> str | None:
    if policy_level not in ALLOWED_BASELINE_PANEL_LEVELS:
        return (
            f"unsupported baseline_policy_level {policy_level}; expected one of "
            f"{list(ALLOWED_BASELINE_PANEL_LEVELS)}"
        )
    allowed = set(BASELINE_PANEL_FULL)
    panel_set = set(panel_ids)
    unknown = sorted(panel_set - allowed)
    if unknown:
        return f"baseline panel contains unsupported ids: {unknown}"

    required = set(BASELINE_PANEL_FULL if policy_level == "full" else BASELINE_PANEL_CORE)
    missing = sorted(required - panel_set)
    if missing:
        return f"baseline panel missing required ids for {policy_level} policy: {missing}"

    if policy_level == "full" and panel_set != allowed:
        extras = sorted(panel_set - allowed)
        if extras:
            return f"full baseline policy forbids extra ids: {extras}"
    return None


def renderer_policy_message(
    *,
    renderer_track: str,
    renderer_profile_id: str,
) -> str | None:
    track = str(renderer_track).strip()
    profile = str(renderer_profile_id).strip()
    if track not in ALLOWED_RENDERER_TRACKS:
        return (
            f"unsupported renderer_track {track}; "
            f"expected one of {list(ALLOWED_RENDERER_TRACKS)}"
        )
    expected = renderer_profile_for_track(track)
    if profile != expected:
        return (
            f"renderer_profile_id {profile} must match {expected} "
            f"for renderer_track {track}"
        )
    return None


def parse_split_ratio_arg(text: str) -> dict[str, float]:
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


def split_policy_report(
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


def manifest_instance_sets_by_split(
    manifest: dict[str, object],
) -> dict[str, set[str]]:
    entries = manifest.get("instances", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.instances must be a list")
    result: dict[str, set[str]] = defaultdict(set)
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest.instances[{idx}] must be object")
        instance_id = str(entry.get("instance_id", "")).strip()
        split_id = str(entry.get("split_id", "")).strip()
        if not instance_id:
            raise ValueError(f"manifest.instances[{idx}] missing instance_id")
        if not split_id:
            raise ValueError(f"manifest.instances[{idx}] missing split_id")
        result[split_id].add(instance_id)
    return result


def release_rotation_report(
    *,
    current_manifest: dict[str, object],
    previous_manifest: dict[str, object] | None,
    private_split_id: str,
    min_public_novelty_ratio: float,
) -> tuple[dict[str, object], bool]:
    current_sets = manifest_instance_sets_by_split(current_manifest)
    public_split_ids = [s for s in OFFICIAL_SPLITS if s != private_split_id]
    current_public = set().union(*(current_sets.get(s, set()) for s in public_split_ids))

    if previous_manifest is None:
        report = {
            "rotation_policy_version": ROTATION_POLICY_VERSION,
            "previous_manifest_present": False,
            "public_split_ids": list(public_split_ids),
            "private_split_id": private_split_id,
            "current_public_count": len(current_public),
            "min_public_novelty_ratio": float(min_public_novelty_ratio),
            "public_novelty_ratio": 1.0 if current_public else 0.0,
            "public_novelty_pass": True,
            "private_to_public_overlap_count": 0,
            "private_to_public_overlap_preview": [],
            "private_to_public_pass": True,
            "status": "ok",
            "error_type": "",
            "note": "rotation checks skipped (no previous manifest provided)",
        }
        return report, True

    prev_sets = manifest_instance_sets_by_split(previous_manifest)
    prev_public = set().union(*(prev_sets.get(s, set()) for s in public_split_ids))
    prev_private = set(prev_sets.get(private_split_id, set()))

    public_novel = current_public - prev_public
    novelty_ratio = (len(public_novel) / float(len(current_public))) if current_public else 0.0
    public_novelty_pass = novelty_ratio >= float(min_public_novelty_ratio)

    private_to_public_overlap = sorted(current_public & prev_private)
    private_to_public_pass = len(private_to_public_overlap) == 0

    passed = bool(public_novelty_pass and private_to_public_pass)
    report = {
        "rotation_policy_version": ROTATION_POLICY_VERSION,
        "previous_manifest_present": True,
        "public_split_ids": list(public_split_ids),
        "private_split_id": private_split_id,
        "current_public_count": len(current_public),
        "previous_public_count": len(prev_public),
        "previous_private_count": len(prev_private),
        "public_novel_count": len(public_novel),
        "min_public_novelty_ratio": float(min_public_novelty_ratio),
        "public_novelty_ratio": novelty_ratio,
        "public_novelty_pass": bool(public_novelty_pass),
        "private_to_public_overlap_count": len(private_to_public_overlap),
        "private_to_public_overlap_preview": private_to_public_overlap[:20],
        "private_to_public_pass": bool(private_to_public_pass),
        "status": "ok" if passed else "error",
        "error_type": "" if passed else "release_rotation_policy_violation",
    }
    return report, passed


def build_report_payload(
    *,
    rows: list[dict[str, object]],
    strict_mode: bool,
    coverage_payload: dict[str, object] | None,
) -> dict[str, object]:
    groups: dict[
        tuple[str, str, str, str, str, bool, str, int, str, str, str],
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
            str(row.get("renderer_profile_id", "unknown")),
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
            renderer_profile_id,
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
                "renderer_profile_id": renderer_profile_id,
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


def panel_ids(panel_arg: str) -> list[str]:
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


def track_for_agent_id(agent_id: str) -> str:
    if agent_id in {"tool", "bl-03", "bl-03-toolplanner"}:
        return "EVAL-TA"
    if agent_id in {"oracle", "bl-04", "bl-04-exactoracle"}:
        return "EVAL-OC"
    return "EVAL-CB"
