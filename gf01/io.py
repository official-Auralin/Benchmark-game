"""
File-format adapters for loading instances and writing run artifacts.

This module translates between external JSON/JSONL files and internal typed
objects. It keeps serialization consistent so experiments can be replayed,
shared, and audited across different environments.
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

import json
from pathlib import Path
from typing import Any

from .meta import (
    ALLOWED_EVAL_TRACKS,
    ALLOWED_MODES,
    ALLOWED_PLAY_PROTOCOLS,
    ALLOWED_TOOL_ALLOWLISTS_BY_TRACK,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    DEFAULT_TOOL_ALLOWLIST_BY_TRACK,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_RUN_FIELDS,
    RUN_RECORD_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    stable_hash_json,
)
from .models import GF01Instance, InterventionAtom, MealyAutomaton, RunRecord


def _track_tool_policy_error(
    eval_track: str,
    tool_allowlist_id: str,
    tool_log_hash: str,
) -> str | None:
    allowlist = str(tool_allowlist_id).strip()
    tool_hash = str(tool_log_hash).strip()
    if eval_track not in ALLOWED_TOOL_ALLOWLISTS_BY_TRACK:
        return None
    if eval_track == "EVAL-CB":
        if allowlist.lower() != "none":
            return "EVAL-CB requires tool_allowlist_id=none"
        if tool_hash:
            return "EVAL-CB requires empty tool_log_hash"
        return None
    allowed = ALLOWED_TOOL_ALLOWLISTS_BY_TRACK[eval_track]
    if allowlist not in allowed:
        return (
            f"{eval_track} requires tool_allowlist_id in {list(allowed)} "
            f"(received {allowlist or '<empty>'})"
        )
    if not tool_hash or tool_hash.lower() == "unknown":
        return f"{eval_track} requires non-empty tool_log_hash"
    return None


def _to_int_valuation(data: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in data.items()}


def automaton_from_dict(data: dict[str, Any]) -> MealyAutomaton:
    transitions: dict[str, dict[str, tuple[str, dict[str, int]]]] = {}
    for state, edges in data["transitions"].items():
        transitions[state] = {}
        for key, payload in edges.items():
            transitions[state][key] = (
                str(payload["next_state"]),
                _to_int_valuation(payload["output"]),
            )
    return MealyAutomaton(
        states=[str(s) for s in data["states"]],
        initial_state=str(data["initial_state"]),
        input_aps=[str(a) for a in data["input_aps"]],
        output_aps=[str(a) for a in data["output_aps"]],
        transitions=transitions,
    )


def instance_from_dict(data: dict[str, Any]) -> GF01Instance:
    automaton = automaton_from_dict(data["automaton"])
    base_trace = [_to_int_valuation(step) for step in data["base_trace"]]
    return GF01Instance(
        instance_id=str(data["instance_id"]),
        automaton=automaton,
        base_trace=base_trace,
        effect_ap=str(data["effect_ap"]),
        t_star=int(data["t_star"]),
        mode=str(data["mode"]),
        window_size=int(data["window_size"]),
        budget_timestep=int(data["budget_timestep"]),
        budget_atoms=int(data["budget_atoms"]),
        seed=int(data["seed"]),
        complexity={str(k): float(v) for k, v in data.get("complexity", {}).items()},
        split_id=str(data.get("split_id", "public_dev")),
        renderer_track=str(data.get("renderer_track", "json")),
    )


def load_instance_bundle(path: str) -> tuple[list[GF01Instance], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bundle_meta: dict[str, Any] = {}
    if isinstance(payload, list):
        instances_raw = payload
    elif isinstance(payload, dict):
        if "instances" in payload:
            instances_raw = payload["instances"]
            bundle_meta = {str(k): v for k, v in payload.items() if k != "instances"}
        elif "instance_id" in payload and "automaton" in payload:
            instances_raw = [payload]
        else:
            raise ValueError(
                "unsupported instance JSON format: expected list, single instance, "
                "or bundle with 'instances' field"
            )
    else:
        raise ValueError("instance JSON payload must be a list or object")
    if not isinstance(instances_raw, list):
        raise ValueError("instances must be a list")
    return [instance_from_dict(item) for item in instances_raw], bundle_meta


def load_instances_json(path: str) -> list[GF01Instance]:
    instances, _ = load_instance_bundle(path)
    return instances


def build_split_manifest(
    instances: list[GF01Instance],
    bundle_meta: dict[str, Any] | None = None,
    source_path: str = "",
) -> dict[str, Any]:
    bundle_meta = bundle_meta or {}
    entries = []
    for inst in sorted(instances, key=lambda i: i.instance_id):
        entries.append(
            {
                "instance_id": inst.instance_id,
                "split_id": inst.split_id,
                "mode": inst.mode,
                "seed": int(inst.seed),
                "t_star": int(inst.t_star),
                "window_size": int(inst.window_size),
                "budget_timestep": int(inst.budget_timestep),
                "budget_atoms": int(inst.budget_atoms),
            }
        )

    split_mode_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (str(entry["split_id"]), str(entry["mode"]))
        split_mode_counts[key] = split_mode_counts.get(key, 0) + 1

    counts = []
    for key, value in sorted(split_mode_counts.items()):
        split_id, mode = key
        counts.append({"split_id": split_id, "mode": mode, "count": int(value)})

    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": bundle_meta.get("benchmark_version", BENCHMARK_VERSION),
        "generator_version": bundle_meta.get("generator_version", GENERATOR_VERSION),
        "checker_version": bundle_meta.get("checker_version", CHECKER_VERSION),
        "harness_version": bundle_meta.get("harness_version", HARNESS_VERSION),
        "source_bundle_schema_version": bundle_meta.get("schema_version", "legacy-instance-list"),
        "source_bundle_hash": bundle_meta.get("instances_hash", stable_hash_json(entries)),
        "source_path": source_path,
        "instance_count": len(entries),
        "group_counts": counts,
        "instances": entries,
    }


def run_record_to_dict(
    record: RunRecord,
    instance: GF01Instance | None = None,
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_meta = run_meta or {}
    out = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "family_id": run_meta.get("family_id", FAMILY_ID),
        "benchmark_version": run_meta.get("benchmark_version", BENCHMARK_VERSION),
        "generator_version": run_meta.get("generator_version", GENERATOR_VERSION),
        "checker_version": run_meta.get("checker_version", CHECKER_VERSION),
        "harness_version": run_meta.get("harness_version", HARNESS_VERSION),
        "git_commit": run_meta.get("git_commit", "unknown"),
        "config_hash": run_meta.get("config_hash", "unknown"),
        "tool_allowlist_id": run_meta.get("tool_allowlist_id", ""),
        "tool_log_hash": run_meta.get("tool_log_hash", ""),
        "play_protocol": run_meta.get("play_protocol", "commit_only"),
        "scored_commit_episode": bool(run_meta.get("scored_commit_episode", True)),
        "instance_id": record.instance_id,
        "eval_track": record.eval_track,
        "renderer_track": record.renderer_track,
        "agent_name": record.agent_name,
        "certificate": [atom.to_tuple() for atom in record.certificate],
        "suff": bool(record.suff),
        "min1": bool(record.min1),
        "valid": bool(record.valid),
        "goal": bool(record.goal),
        "eff_t": int(record.eff_t),
        "eff_a": int(record.eff_a),
        "ap_precision": float(record.ap_precision),
        "ap_recall": float(record.ap_recall),
        "ap_f1": float(record.ap_f1),
        "ts_precision": float(record.ts_precision),
        "ts_recall": float(record.ts_recall),
        "ts_f1": float(record.ts_f1),
        "diagnostic_status": record.diagnostic_status,
        "diagnostic_runtime_ms": int(record.diagnostic_runtime_ms),
    }
    if instance is not None:
        out.update(
            {
                "split_id": instance.split_id,
                "mode": instance.mode,
                "t_star": int(instance.t_star),
                "window_size": int(instance.window_size),
                "budget_timestep": int(instance.budget_timestep),
                "budget_atoms": int(instance.budget_atoms),
                "seed": int(instance.seed),
                "complexity": {k: float(instance.complexity[k]) for k in sorted(instance.complexity)},
            }
        )
    return out


def write_run_records_jsonl(
    path: str,
    records: list[RunRecord],
    instance_lookup: dict[str, GF01Instance] | None = None,
    run_meta: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    for record in records:
        inst = instance_lookup.get(record.instance_id) if instance_lookup else None
        lines.append(json.dumps(run_record_to_dict(record, inst, run_meta=run_meta), sort_keys=True))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return {str(k): v for k, v in payload.items()}


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _missing_fields(payload: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [field for field in required if field not in payload]


def validate_run_rows(rows: list[dict[str, Any]], strict: bool = False) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows):
        missing = _missing_fields(row, REQUIRED_RUN_FIELDS)
        if missing:
            errors.append(f"row {idx}: missing required fields: {', '.join(missing)}")
            continue

        if strict:
            if str(row.get("schema_version")) != RUN_RECORD_SCHEMA_VERSION:
                errors.append(
                    f"row {idx}: schema_version={row.get('schema_version')} "
                    f"(expected {RUN_RECORD_SCHEMA_VERSION})"
                )
            if str(row.get("family_id")) != FAMILY_ID:
                errors.append(
                    f"row {idx}: family_id={row.get('family_id')} (expected {FAMILY_ID})"
                )

        eval_track = str(row.get("eval_track"))
        if eval_track not in ALLOWED_EVAL_TRACKS:
            errors.append(
                f"row {idx}: eval_track={eval_track} not in {list(ALLOWED_EVAL_TRACKS)}"
            )
        else:
            tool_policy_error = _track_tool_policy_error(
                eval_track=eval_track,
                tool_allowlist_id=str(row.get("tool_allowlist_id", "")),
                tool_log_hash=str(row.get("tool_log_hash", "")),
            )
            if tool_policy_error is not None:
                errors.append(f"row {idx}: {tool_policy_error}")

        mode = str(row.get("mode"))
        if mode not in ALLOWED_MODES:
            errors.append(f"row {idx}: mode={mode} not in {list(ALLOWED_MODES)}")

        play_protocol = str(row.get("play_protocol"))
        if play_protocol not in ALLOWED_PLAY_PROTOCOLS:
            errors.append(
                f"row {idx}: play_protocol={play_protocol} "
                f"not in {list(ALLOWED_PLAY_PROTOCOLS)}"
            )

        if not isinstance(row.get("certificate"), list):
            errors.append(f"row {idx}: certificate must be a list")

        for bool_field in ("suff", "min1", "valid", "goal", "scored_commit_episode"):
            if not isinstance(row.get(bool_field), bool):
                errors.append(f"row {idx}: {bool_field} must be boolean")

        for float_field in ("ap_f1", "ts_f1"):
            try:
                float(row.get(float_field))
            except Exception:
                errors.append(f"row {idx}: {float_field} must be numeric")

        if strict:
            for meta_field in (
                "benchmark_version",
                "generator_version",
                "checker_version",
                "harness_version",
                "git_commit",
                "config_hash",
                "tool_allowlist_id",
                "play_protocol",
                "renderer_track",
                "split_id",
                "agent_name",
            ):
                value = str(row.get(meta_field, "")).strip()
                if not value or value == "unknown":
                    errors.append(f"row {idx}: {meta_field} must be populated in strict mode")
    return errors


def validate_manifest(manifest: dict[str, Any], strict: bool = False) -> list[str]:
    errors: list[str] = []
    missing = _missing_fields(manifest, REQUIRED_MANIFEST_FIELDS)
    if missing:
        errors.append(f"manifest: missing required fields: {', '.join(missing)}")
        return errors

    if strict and str(manifest.get("schema_version")) != SPLIT_MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"manifest: schema_version={manifest.get('schema_version')} "
            f"(expected {SPLIT_MANIFEST_SCHEMA_VERSION})"
        )
    if strict and str(manifest.get("family_id")) != FAMILY_ID:
        errors.append(
            f"manifest: family_id={manifest.get('family_id')} (expected {FAMILY_ID})"
        )

    entries = manifest.get("instances")
    if not isinstance(entries, list):
        errors.append("manifest: instances must be a list")
        return errors

    if int(manifest.get("instance_count", -1)) != len(entries):
        errors.append(
            f"manifest: instance_count={manifest.get('instance_count')} "
            f"does not match len(instances)={len(entries)}"
        )

    seen_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"manifest.instances[{idx}]: must be object")
            continue
        for field in ("instance_id", "split_id", "mode"):
            if field not in entry or str(entry[field]).strip() == "":
                errors.append(f"manifest.instances[{idx}]: missing {field}")
        if "instance_id" in entry:
            ident = str(entry["instance_id"])
            if ident in seen_ids:
                errors.append(f"manifest.instances[{idx}]: duplicate instance_id={ident}")
            seen_ids.add(ident)
        if "mode" in entry and str(entry["mode"]) not in ALLOWED_MODES:
            errors.append(
                f"manifest.instances[{idx}]: mode={entry['mode']} "
                f"not in {list(ALLOWED_MODES)}"
            )
    return errors


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sanitize_certificate(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def migrate_run_rows(
    rows: list[dict[str, Any]],
    defaults: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = manifest or {}
    manifest_entries = manifest.get("instances", [])
    manifest_index: dict[str, dict[str, Any]] = {}
    if isinstance(manifest_entries, list):
        for entry in manifest_entries:
            if isinstance(entry, dict) and "instance_id" in entry:
                manifest_index[str(entry["instance_id"])] = entry

    fill_counts: dict[str, int] = {}
    coercion_counts: dict[str, int] = {}
    migrated: list[dict[str, Any]] = []

    def _mark(counter: dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    def _fill(
        row: dict[str, Any],
        key: str,
        value: Any,
        *,
        treat_unknown_as_missing: bool = True,
    ) -> None:
        if value is None:
            return
        current = row.get(key)
        missing = key not in row or current is None or str(current).strip() == ""
        if treat_unknown_as_missing and str(current).strip().lower() == "unknown":
            missing = True
        if missing:
            row[key] = value
            _mark(fill_counts, key)

    for idx, row in enumerate(rows):
        item = {str(k): v for k, v in row.items()}

        instance_id = str(item.get("instance_id", "")).strip()
        if not instance_id:
            instance_id = f"legacy-row-{idx:06d}"
            item["instance_id"] = instance_id
            _mark(fill_counts, "instance_id")
        manifest_entry = manifest_index.get(instance_id, {})

        _fill(item, "schema_version", defaults.get("schema_version", RUN_RECORD_SCHEMA_VERSION))
        _fill(item, "family_id", defaults.get("family_id", FAMILY_ID))
        _fill(item, "benchmark_version", defaults.get("benchmark_version"))
        _fill(item, "generator_version", defaults.get("generator_version"))
        _fill(item, "checker_version", defaults.get("checker_version"))
        _fill(item, "harness_version", defaults.get("harness_version"))
        _fill(item, "git_commit", defaults.get("git_commit"))
        _fill(item, "config_hash", defaults.get("config_hash"))
        _fill(item, "tool_allowlist_id", defaults.get("tool_allowlist_id", "none"))
        _fill(item, "tool_log_hash", defaults.get("tool_log_hash", ""), treat_unknown_as_missing=False)
        _fill(item, "play_protocol", defaults.get("play_protocol", "commit_only"))
        _fill(
            item,
            "scored_commit_episode",
            defaults.get("scored_commit_episode", True),
            treat_unknown_as_missing=False,
        )

        _fill(item, "eval_track", defaults.get("eval_track", "EVAL-CB"))
        _fill(item, "renderer_track", defaults.get("renderer_track", "json"))
        _fill(item, "agent_name", defaults.get("agent_name", "legacy-agent"))

        _fill(item, "split_id", manifest_entry.get("split_id"))
        _fill(item, "mode", manifest_entry.get("mode"))
        _fill(item, "seed", manifest_entry.get("seed"))
        _fill(item, "t_star", manifest_entry.get("t_star"))
        _fill(item, "window_size", manifest_entry.get("window_size"))
        _fill(item, "budget_timestep", manifest_entry.get("budget_timestep"))
        _fill(item, "budget_atoms", manifest_entry.get("budget_atoms"))

        _fill(item, "split_id", defaults.get("split_id", "public_dev"))
        _fill(item, "mode", defaults.get("mode", "normal"))
        _fill(item, "seed", defaults.get("seed", 0))

        if str(item.get("eval_track", "")).strip() not in ALLOWED_EVAL_TRACKS:
            item["eval_track"] = defaults.get("eval_track", "EVAL-CB")
            _mark(coercion_counts, "eval_track")
        if str(item.get("mode", "")).strip() not in ALLOWED_MODES:
            item["mode"] = defaults.get("mode", "normal")
            _mark(coercion_counts, "mode")
        if str(item.get("play_protocol", "")).strip() not in ALLOWED_PLAY_PROTOCOLS:
            item["play_protocol"] = defaults.get("play_protocol", "commit_only")
            _mark(coercion_counts, "play_protocol")

        eval_track = str(item.get("eval_track", "EVAL-CB")).strip()
        allowlist = str(item.get("tool_allowlist_id", "")).strip()
        tool_hash = str(item.get("tool_log_hash", "")).strip()
        if eval_track == "EVAL-CB":
            if allowlist.lower() != "none":
                item["tool_allowlist_id"] = "none"
                _mark(coercion_counts, "tool_allowlist_id")
            if tool_hash:
                item["tool_log_hash"] = ""
                _mark(coercion_counts, "tool_log_hash")
        elif eval_track in {"EVAL-TA", "EVAL-OC"}:
            allowed = ALLOWED_TOOL_ALLOWLISTS_BY_TRACK[eval_track]
            default_allowlist = DEFAULT_TOOL_ALLOWLIST_BY_TRACK[eval_track]
            if allowlist not in allowed:
                item["tool_allowlist_id"] = default_allowlist
                _mark(coercion_counts, "tool_allowlist_id")
            if not tool_hash or tool_hash.lower() == "unknown":
                item["tool_log_hash"] = stable_hash_json(
                    {
                        "migration_row": idx,
                        "instance_id": item.get("instance_id", ""),
                        "eval_track": eval_track,
                        "tool_allowlist_id": item.get("tool_allowlist_id", default_allowlist),
                    }
                )[:16]
                _mark(coercion_counts, "tool_log_hash")

        cert_before = item.get("certificate")
        cert_after = _sanitize_certificate(cert_before)
        item["certificate"] = cert_after
        if cert_before is not cert_after and cert_before != cert_after:
            _mark(coercion_counts, "certificate")

        for bool_field in ("suff", "min1", "valid", "goal", "scored_commit_episode"):
            before = item.get(bool_field, False)
            after = _to_bool(before)
            item[bool_field] = after
            if before is not after and before != after:
                _mark(coercion_counts, bool_field)

        for float_field in (
            "ap_precision",
            "ap_recall",
            "ap_f1",
            "ts_precision",
            "ts_recall",
            "ts_f1",
        ):
            before = item.get(float_field, 0.0)
            after = _to_float(before, default=0.0)
            item[float_field] = after
            if before != after:
                _mark(coercion_counts, float_field)

        for int_field in (
            "eff_t",
            "eff_a",
            "diagnostic_runtime_ms",
            "seed",
            "t_star",
            "window_size",
            "budget_timestep",
            "budget_atoms",
        ):
            if int_field not in item:
                continue
            before = item[int_field]
            try:
                after = int(before)
                item[int_field] = after
                if before != after:
                    _mark(coercion_counts, int_field)
            except Exception:
                pass

        _fill(item, "diagnostic_status", "not-run", treat_unknown_as_missing=False)
        _fill(item, "diagnostic_runtime_ms", 0, treat_unknown_as_missing=False)
        _fill(item, "goal", False, treat_unknown_as_missing=False)
        _fill(item, "suff", False, treat_unknown_as_missing=False)
        _fill(item, "min1", False, treat_unknown_as_missing=False)
        _fill(item, "valid", False, treat_unknown_as_missing=False)
        _fill(item, "play_protocol", "commit_only", treat_unknown_as_missing=False)
        _fill(item, "scored_commit_episode", True, treat_unknown_as_missing=False)
        _fill(item, "ap_f1", 0.0, treat_unknown_as_missing=False)
        _fill(item, "ts_f1", 0.0, treat_unknown_as_missing=False)

        migrated.append(item)

    stats = {
        "input_rows": len(rows),
        "output_rows": len(migrated),
        "filled_fields": fill_counts,
        "coerced_fields": coercion_counts,
        "manifest_join_hits": sum(1 for r in migrated if str(r.get("instance_id")) in manifest_index),
    }
    return migrated, stats
