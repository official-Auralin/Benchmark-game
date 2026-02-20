"""
Q-033 high-performance sweep utilities for closure-ready profiling evidence.

This module implements deterministic helpers for building stratified seed
manifests, running one profiling sweep replicate, and checking closure rules
across replicates. It operationalizes the pre-registered Q-033 protocol.
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

import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import date
from typing import Any

from .baselines import make_agent
from .checks import evaluate_suite, run_priority_h3_checks
from .generator import generate_instance, generate_suite
from .meta import (
    BASELINE_PANEL_FULL,
    BENCHMARK_VERSION,
    CHECKER_VERSION,
    FAMILY_ID,
    GENERATOR_VERSION,
    HARNESS_VERSION,
    config_hash,
    current_git_commit,
)
from .models import GeneratorConfig, GF01Instance
from .profiling import PerformanceGates
from .verifier import find_valid_certificates


Q033_PROTOCOL_VERSION = "gf01.q033_protocol.v1"
Q033_MANIFEST_SCHEMA_VERSION = "gf01.q033_manifest.v1"
Q033_SWEEP_SCHEMA_VERSION = "gf01.q033_sweep.v1"
Q033_CLOSURE_SCHEMA_VERSION = "gf01.q033_closure.v1"
Q033_QUARTILES = ("Q1", "Q2", "Q3", "Q4")


def _track_for_agent_id(agent_id: str) -> str:
    token = str(agent_id).strip().lower()
    if token == "tool":
        return "EVAL-TA"
    if token == "oracle":
        return "EVAL-OC"
    return "EVAL-CB"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _instance_complexity_score(instance: GF01Instance) -> float:
    complexity = instance.complexity or {}
    values: list[float] = []
    for value in complexity.values():
        try:
            values.append(float(value))
        except Exception:
            continue
    return _mean(values)


def _assign_instance_quartiles(instances: list[GF01Instance]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        rows.append(
            {
                "instance_id": instance.instance_id,
                "seed": int(instance.seed),
                "complexity_score": float(_instance_complexity_score(instance)),
            }
        )
    rows.sort(key=lambda row: (float(row["complexity_score"]), str(row["instance_id"])))
    n = len(rows)
    if n == 0:
        return rows
    for idx, row in enumerate(rows):
        quartile_idx = min(4, (idx * 4) // n + 1)
        row["quartile"] = f"Q{quartile_idx}"
    return rows


def build_q033_manifests(
    *,
    seed_start: int,
    candidate_count: int,
    replicates: int,
    per_quartile: int,
    split_id: str,
    cfg: GeneratorConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or GeneratorConfig()
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if per_quartile <= 0:
        raise ValueError("per_quartile must be positive")

    max_seed_span = max(int(candidate_count * 10), int(candidate_count + 50))
    accepted_instances: list[GF01Instance] = []
    accepted_seeds: list[int] = []
    skipped_seeds: list[int] = []
    cursor = int(seed_start)
    while len(accepted_instances) < candidate_count and (cursor - seed_start) < max_seed_span:
        try:
            instance, _ = generate_instance(seed=cursor, cfg=cfg, split_id=split_id)
            accepted_instances.append(instance)
            accepted_seeds.append(int(cursor))
        except Exception:
            skipped_seeds.append(int(cursor))
        cursor += 1

    if len(accepted_instances) < candidate_count:
        raise RuntimeError(
            "insufficient accepted instances while scanning seed range: "
            f"requested={candidate_count}, accepted={len(accepted_instances)}, "
            f"seed_start={seed_start}, scanned={cursor - seed_start}, "
            f"max_seed_span={max_seed_span}"
        )

    instances = accepted_instances
    quartiled = _assign_instance_quartiles(instances)

    bucket: dict[str, list[dict[str, Any]]] = {q: [] for q in Q033_QUARTILES}
    for row in quartiled:
        quartile = str(row.get("quartile", ""))
        if quartile in bucket:
            bucket[quartile].append(row)

    required_per_quartile = int(replicates * per_quartile)
    available_counts = {q: len(bucket[q]) for q in Q033_QUARTILES}
    insufficient = {
        q: {"required": required_per_quartile, "available": available_counts[q]}
        for q in Q033_QUARTILES
        if available_counts[q] < required_per_quartile
    }
    if insufficient:
        raise RuntimeError(
            "insufficient candidate seeds for requested balanced manifests: "
            f"{insufficient}"
        )

    manifests: list[dict[str, Any]] = []
    for rep_idx in range(replicates):
        start = rep_idx * per_quartile
        end = start + per_quartile
        quartile_rows: dict[str, list[dict[str, Any]]] = {}
        seeds_flat: list[int] = []
        for quartile in Q033_QUARTILES:
            selected = bucket[quartile][start:end]
            quartile_rows[quartile] = [
                {
                    "seed": int(row["seed"]),
                    "instance_id": str(row["instance_id"]),
                    "complexity_score": float(row["complexity_score"]),
                }
                for row in selected
            ]
            seeds_flat.extend(int(row["seed"]) for row in selected)

        manifest = {
            "schema_version": Q033_MANIFEST_SCHEMA_VERSION,
            "protocol_version": Q033_PROTOCOL_VERSION,
            "manifest_id": f"q033-rep-{rep_idx + 1:02d}",
            "replicate_index": rep_idx + 1,
            "replicate_count": int(replicates),
            "per_quartile": int(per_quartile),
            "split_id": str(split_id),
            "seed_start": int(seed_start),
            "candidate_count": int(candidate_count),
            "accepted_seed_start": int(accepted_seeds[0]) if accepted_seeds else int(seed_start),
            "accepted_seed_end": int(accepted_seeds[-1]) if accepted_seeds else int(seed_start),
            "seed_count": len(seeds_flat),
            "seeds": seeds_flat,
            "quartile_rows": quartile_rows,
            "generator_config": asdict(cfg),
            "generator_config_hash": config_hash(cfg),
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": GENERATOR_VERSION,
            "checker_version": CHECKER_VERSION,
            "harness_version": HARNESS_VERSION,
            "git_commit": current_git_commit(),
            "generated_on": date.today().isoformat(),
            "quartile_assignment_method": "global_complexity_sort_index_quartiles",
        }
        manifests.append(manifest)

    all_seed_sets = [set(manifest["seeds"]) for manifest in manifests]
    overlaps: list[dict[str, Any]] = []
    for i in range(len(all_seed_sets)):
        for j in range(i + 1, len(all_seed_sets)):
            overlap = sorted(all_seed_sets[i].intersection(all_seed_sets[j]))
            overlaps.append(
                {
                    "replicate_i": i + 1,
                    "replicate_j": j + 1,
                    "overlap_count": len(overlap),
                    "overlap_preview": overlap[:20],
                }
            )

    return {
        "status": "ok",
        "schema_version": Q033_MANIFEST_SCHEMA_VERSION,
        "protocol_version": Q033_PROTOCOL_VERSION,
        "seed_start": int(seed_start),
        "candidate_count": int(candidate_count),
        "accepted_seed_start": int(accepted_seeds[0]) if accepted_seeds else int(seed_start),
        "accepted_seed_end": int(accepted_seeds[-1]) if accepted_seeds else int(seed_start),
        "skipped_seed_count": len(skipped_seeds),
        "skipped_seed_preview": skipped_seeds[:20],
        "replicate_count": int(replicates),
        "per_quartile": int(per_quartile),
        "required_per_quartile": int(required_per_quartile),
        "available_counts": available_counts,
        "manifest_seed_overlaps": overlaps,
        "manifests": manifests,
    }


def _hardware_declaration() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count_logical": int(os.cpu_count() or 0),
        "executable": sys.executable,
    }


def _quartile_rows_from_records(
    records,
    quartile_by_instance: dict[str, str],
) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {q: [] for q in Q033_QUARTILES}
    for record in records:
        quartile = quartile_by_instance.get(str(record.instance_id), "")
        if quartile in grouped:
            grouped[quartile].append(record)
    return grouped


def run_q033_sweep(
    *,
    manifest: dict[str, Any],
    panel: list[str],
    cfg: GeneratorConfig | None = None,
    gates: PerformanceGates | None = None,
    seed_base: int = 0,
    max_quartile_truncation_rate: float = 0.30,
    min_quartile_oracle_minus_greedy_gap: float = 0.05,
    max_quartile_runtime_gate_failures: int = 1,
) -> dict[str, Any]:
    cfg = cfg or GeneratorConfig()
    gates = gates or PerformanceGates()
    seeds_raw = manifest.get("seeds", [])
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("manifest.seeds must be a non-empty list")
    seeds = [int(seed) for seed in seeds_raw]
    split_id = str(manifest.get("split_id", "q033_internal"))

    t_gen_start = time.perf_counter()
    instances = generate_suite(seeds=seeds, split_id=split_id, cfg=cfg)
    t_gen_end = time.perf_counter()
    generate_ms_mean = (t_gen_end - t_gen_start) * 1000.0 / float(max(1, len(instances)))

    quartiled = _assign_instance_quartiles(instances)
    quartile_by_instance = {str(row["instance_id"]): str(row["quartile"]) for row in quartiled}
    quartile_counts: dict[str, int] = {q: 0 for q in Q033_QUARTILES}
    for row in quartiled:
        quartile_counts[str(row["quartile"])] += 1

    minset_times_ms: list[float] = []
    minset_times_by_quartile: dict[str, list[float]] = {q: [] for q in Q033_QUARTILES}
    minset_nodes: list[int] = []
    truncation_count = 0
    truncation_by_quartile: dict[str, int] = {q: 0 for q in Q033_QUARTILES}
    minimal_cache: dict[str, list[list[Any]]] = {}

    for instance in instances:
        q = quartile_by_instance.get(instance.instance_id, "Q4")
        t0 = time.perf_counter()
        search = find_valid_certificates(instance, max_nodes=cfg.max_exact_nodes, max_results=32)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        minset_times_ms.append(elapsed_ms)
        minset_times_by_quartile[q].append(elapsed_ms)
        minset_nodes.append(int(search.nodes_visited))
        trunc = int(search.truncated)
        truncation_count += trunc
        truncation_by_quartile[q] += trunc
        minimal_cache[instance.instance_id] = search.certificates

    minset_ms_mean = _mean(minset_times_ms)
    truncation_rate = truncation_count / float(max(1, len(instances)))

    panel_summary: dict[str, Any] = {}
    panel_records: dict[str, list[Any]] = {}
    eval_durations_ms: dict[str, float] = {}

    for idx, agent_id in enumerate(panel):
        token = str(agent_id).strip().lower()
        agent = make_agent(token)
        eval_track = _track_for_agent_id(token)
        t_eval_0 = time.perf_counter()
        records, aggregate = evaluate_suite(
            agent=agent,
            instances=instances,
            eval_track=eval_track,
            renderer_track="json",
            seed=int(seed_base + idx),
            minimal_cache=minimal_cache,
        )
        t_eval_1 = time.perf_counter()
        elapsed_ms = (t_eval_1 - t_eval_0) * 1000.0
        eval_durations_ms[token] = elapsed_ms
        panel_records[token] = records

        quartile_records = _quartile_rows_from_records(records, quartile_by_instance)
        quartile_metrics: dict[str, Any] = {}
        for quartile in Q033_QUARTILES:
            bucket = quartile_records[quartile]
            count = len(bucket)
            goal_rate = (
                sum(1 for rec in bucket if bool(rec.goal)) / float(max(1, count))
            )
            certified_rate = (
                sum(1 for rec in bucket if bool(rec.valid)) / float(max(1, count))
            )
            quartile_metrics[quartile] = {
                "count": int(count),
                "goal_rate": goal_rate,
                "certified_rate": certified_rate,
            }
        panel_summary[token] = {
            "agent_name": agent.name,
            "eval_track": eval_track,
            "row_count": len(records),
            "duration_ms": elapsed_ms,
            "aggregate": aggregate,
            "quartiles": quartile_metrics,
        }

    if "greedy" not in panel_summary or "oracle" not in panel_summary:
        raise ValueError("panel must include greedy and oracle for Q-033 gates")

    eval_ms_mean = (
        (eval_durations_ms.get("greedy", 0.0) + eval_durations_ms.get("oracle", 0.0))
        / float(max(1, len(instances) * 2))
    )

    t_checks_0 = time.perf_counter()
    checks_payload = run_priority_h3_checks(seed_base=int(seed_base))
    t_checks_1 = time.perf_counter()
    checks_total_ms = (t_checks_1 - t_checks_0) * 1000.0

    greedy_cert = float(panel_summary["greedy"]["aggregate"].get("certified_rate", 0.0))
    oracle_cert = float(panel_summary["oracle"]["aggregate"].get("certified_rate", 0.0))
    oracle_minus_greedy_gap = oracle_cert - greedy_cert

    global_gate_results = {
        "generate_ms_mean_ok": generate_ms_mean <= gates.max_generate_ms_mean,
        "minset_ms_mean_ok": minset_ms_mean <= gates.max_minset_ms_mean,
        "eval_ms_mean_ok": eval_ms_mean <= gates.max_eval_ms_mean,
        "checks_total_ms_ok": checks_total_ms <= gates.max_checks_total_ms,
        "truncation_rate_ok": truncation_rate <= gates.max_truncation_rate,
        "baseline_spread_ok": (
            oracle_minus_greedy_gap >= gates.min_oracle_minus_greedy_certified_gap
        ),
    }

    quartile_gates: dict[str, Any] = {}
    for quartile in Q033_QUARTILES:
        count = int(quartile_counts.get(quartile, 0))
        q_trunc_count = int(truncation_by_quartile.get(quartile, 0))
        q_trunc_rate = q_trunc_count / float(max(1, count))
        greedy_q = float(panel_summary["greedy"]["quartiles"][quartile]["certified_rate"])
        oracle_q = float(panel_summary["oracle"]["quartiles"][quartile]["certified_rate"])
        q_gap = oracle_q - greedy_q
        q_minset_mean = _mean(minset_times_by_quartile.get(quartile, []))

        runtime_failures: list[str] = []
        if not global_gate_results["generate_ms_mean_ok"]:
            runtime_failures.append("generate_ms_mean_ok")
        if q_minset_mean > gates.max_minset_ms_mean:
            runtime_failures.append("minset_ms_mean_ok")
        if not global_gate_results["eval_ms_mean_ok"]:
            runtime_failures.append("eval_ms_mean_ok")
        if not global_gate_results["checks_total_ms_ok"]:
            runtime_failures.append("checks_total_ms_ok")

        q_gate = {
            "count": count,
            "minset_ms_mean": q_minset_mean,
            "truncation_rate": q_trunc_rate,
            "oracle_minus_greedy_certified_gap": q_gap,
            "truncation_rate_ok": q_trunc_rate <= max_quartile_truncation_rate,
            "baseline_gap_ok": q_gap >= min_quartile_oracle_minus_greedy_gap,
            "runtime_gate_failures": runtime_failures,
            "runtime_gate_fail_count": len(runtime_failures),
            "runtime_gate_fail_count_ok": (
                len(runtime_failures) <= int(max_quartile_runtime_gate_failures)
            ),
        }
        q_gate["passed_all"] = bool(
            q_gate["truncation_rate_ok"]
            and q_gate["baseline_gap_ok"]
            and q_gate["runtime_gate_fail_count_ok"]
        )
        quartile_gates[quartile] = q_gate

    global_failed = [key for key, ok in global_gate_results.items() if not ok]
    global_passed_all = len(global_failed) == 0
    stratified_failed_quartiles = [
        quartile for quartile, payload in quartile_gates.items() if not bool(payload["passed_all"])
    ]
    stratified_passed_all = len(stratified_failed_quartiles) == 0

    return {
        "status": "ok" if (global_passed_all and stratified_passed_all) else "error",
        "schema_version": Q033_SWEEP_SCHEMA_VERSION,
        "protocol_version": Q033_PROTOCOL_VERSION,
        "family_id": FAMILY_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "checker_version": CHECKER_VERSION,
        "harness_version": HARNESS_VERSION,
        "git_commit": current_git_commit(),
        "generator_config_hash": config_hash(cfg),
        "generator_config": asdict(cfg),
        "hardware": _hardware_declaration(),
        "seed_manifest": {
            "manifest_id": str(manifest.get("manifest_id", "")),
            "replicate_index": int(manifest.get("replicate_index", 0)),
            "seed_count": len(seeds),
            "seeds": seeds,
        },
        "quartile_assignment": {
            "method": "global_complexity_sort_index_quartiles",
            "rows": quartiled,
            "counts": quartile_counts,
        },
        "panel": panel,
        "timings_ms": {
            "generate_ms_mean": generate_ms_mean,
            "minset_ms_mean": minset_ms_mean,
            "eval_ms_mean": eval_ms_mean,
            "checks_total_ms": checks_total_ms,
        },
        "search_stats": {
            "truncation_rate": truncation_rate,
            "truncation_count": truncation_count,
            "min_nodes": min(minset_nodes) if minset_nodes else 0,
            "max_nodes": max(minset_nodes) if minset_nodes else 0,
            "mean_nodes": _mean([float(node) for node in minset_nodes]),
        },
        "calibration_stats": {
            "greedy_certified_rate": greedy_cert,
            "oracle_certified_rate": oracle_cert,
            "oracle_minus_greedy_certified_gap": oracle_minus_greedy_gap,
        },
        "panel_summary": panel_summary,
        "priority_checks": checks_payload,
        "gates": {
            "global_thresholds": asdict(gates),
            "global_results": global_gate_results,
            "global_failed": global_failed,
            "global_passed_all": global_passed_all,
            "stratified_thresholds": {
                "max_quartile_truncation_rate": float(max_quartile_truncation_rate),
                "min_quartile_oracle_minus_greedy_gap": float(
                    min_quartile_oracle_minus_greedy_gap
                ),
                "max_quartile_runtime_gate_failures": int(max_quartile_runtime_gate_failures),
            },
            "stratified_results": quartile_gates,
            "stratified_failed_quartiles": stratified_failed_quartiles,
            "stratified_passed_all": stratified_passed_all,
            "passed_all": bool(global_passed_all and stratified_passed_all),
        },
    }


def q033_closure_check(
    *,
    sweeps: list[dict[str, Any]],
    require_disjoint_seeds: bool = True,
) -> dict[str, Any]:
    if len(sweeps) < 2:
        raise ValueError("closure check requires at least two sweep payloads")

    replicate_rows: list[dict[str, Any]] = []
    seed_sets: list[set[int]] = []
    all_passed = True
    reasons: list[str] = []

    for idx, sweep in enumerate(sweeps):
        replicate_id = idx + 1
        gates = sweep.get("gates", {})
        passed_all = bool(gates.get("passed_all", False))
        seed_manifest = sweep.get("seed_manifest", {})
        seeds_raw = seed_manifest.get("seeds", [])
        seeds = {int(seed) for seed in seeds_raw} if isinstance(seeds_raw, list) else set()
        seed_sets.append(seeds)
        if not passed_all:
            all_passed = False
            reasons.append(f"replicate_{replicate_id}_failed_gates")
        replicate_rows.append(
            {
                "replicate_index": replicate_id,
                "manifest_id": str(seed_manifest.get("manifest_id", "")),
                "seed_count": len(seeds),
                "passed_all": passed_all,
                "status": str(sweep.get("status", "unknown")),
                "global_failed": list(gates.get("global_failed", [])),
                "stratified_failed_quartiles": list(gates.get("stratified_failed_quartiles", [])),
            }
        )

    overlaps: list[dict[str, Any]] = []
    disjoint_ok = True
    for i in range(len(seed_sets)):
        for j in range(i + 1, len(seed_sets)):
            overlap = sorted(seed_sets[i].intersection(seed_sets[j]))
            if overlap:
                disjoint_ok = False
            overlaps.append(
                {
                    "replicate_i": i + 1,
                    "replicate_j": j + 1,
                    "overlap_count": len(overlap),
                    "overlap_preview": overlap[:20],
                }
            )

    if require_disjoint_seeds and not disjoint_ok:
        all_passed = False
        reasons.append("replicate_seed_sets_overlap")

    close_q033 = bool(all_passed)
    return {
        "status": "ok" if close_q033 else "error",
        "schema_version": Q033_CLOSURE_SCHEMA_VERSION,
        "protocol_version": Q033_PROTOCOL_VERSION,
        "require_disjoint_seeds": bool(require_disjoint_seeds),
        "replicate_count": len(sweeps),
        "replicate_results": replicate_rows,
        "seed_overlaps": overlaps,
        "disjoint_ok": disjoint_ok,
        "close_q033": close_q033,
        "failure_reasons": reasons,
    }
