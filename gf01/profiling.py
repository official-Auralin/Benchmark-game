"""
Profiling and performance-gate evaluation for the GF-01 Python harness.

This module measures generation, verification, evaluation, and check runtimes.
It reports gate pass/fail outcomes so we can decide whether current Python
implementation speed is acceptable or needs further optimization.
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

import statistics
import time
from dataclasses import asdict, dataclass

from .baselines import ExactOracleAgent, GreedyLocalAgent
from .checks import evaluate_suite, run_priority_h3_checks
from .generator import generate_suite
from .models import GF01Instance, GeneratorConfig
from .verifier import find_valid_certificates


@dataclass
class PerformanceGates:
    max_generate_ms_mean: float = 1200.0
    max_minset_ms_mean: float = 2500.0
    max_eval_ms_mean: float = 1500.0
    max_checks_total_ms: float = 30000.0
    max_truncation_rate: float = 0.25
    min_oracle_minus_greedy_certified_gap: float = 0.10


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def profile_pipeline(
    seed_base: int,
    public_count: int,
    private_count: int,
    cfg: GeneratorConfig | None = None,
    gates: PerformanceGates | None = None,
) -> dict[str, object]:
    cfg = cfg or GeneratorConfig()
    gates = gates or PerformanceGates()

    seeds_public = [seed_base + i for i in range(public_count)]
    seeds_private = [seed_base + 100 + i for i in range(private_count)]

    t0 = time.perf_counter()
    t_gen_public_start = time.perf_counter()
    public_instances = generate_suite(seeds_public, split_id="public_dev", cfg=cfg)
    t_gen_public_end = time.perf_counter()

    t_gen_private_start = time.perf_counter()
    private_instances = generate_suite(seeds_private, split_id="private_eval", cfg=cfg)
    t_gen_private_end = time.perf_counter()
    t1 = time.perf_counter()

    all_instances: list[GF01Instance] = public_instances + private_instances

    minset_times: list[float] = []
    minset_nodes: list[int] = []
    trunc_count = 0
    minimal_cache: dict[str, list[list]] = {}
    for inst in all_instances:
        ts = time.perf_counter()
        search = find_valid_certificates(inst, max_nodes=cfg.max_exact_nodes, max_results=32)
        te = time.perf_counter()
        minset_times.append(_ms(ts, te))
        minset_nodes.append(search.nodes_visited)
        trunc_count += int(search.truncated)
        minimal_cache[inst.instance_id] = search.certificates

    t_eval_start = time.perf_counter()
    greedy = GreedyLocalAgent()
    oracle = ExactOracleAgent()
    _, greedy_private = evaluate_suite(
        greedy,
        private_instances,
        eval_track="EVAL-CB",
        renderer_track="json",
        seed=seed_base,
        minimal_cache=minimal_cache,
    )
    _, oracle_private = evaluate_suite(
        oracle,
        private_instances,
        eval_track="EVAL-OC",
        renderer_track="json",
        seed=seed_base,
        minimal_cache=minimal_cache,
    )
    t_eval_end = time.perf_counter()

    t_checks_start = time.perf_counter()
    checks = run_priority_h3_checks(seed_base=seed_base)
    t_checks_end = time.perf_counter()

    generate_ms_mean = _mean(
        [
            _ms(t_gen_public_start, t_gen_public_end) / max(1, len(public_instances)),
            _ms(t_gen_private_start, t_gen_private_end) / max(1, len(private_instances)),
        ]
    )
    minset_ms_mean = _mean(minset_times)
    eval_ms_mean = _ms(t_eval_start, t_eval_end) / max(1, len(private_instances) * 2)
    checks_total_ms = _ms(t_checks_start, t_checks_end)
    trunc_rate = trunc_count / max(1, len(all_instances))
    certified_gap = oracle_private.get("certified_rate", 0.0) - greedy_private.get(
        "certified_rate", 0.0
    )

    gate_results = {
        "generate_ms_mean_ok": generate_ms_mean <= gates.max_generate_ms_mean,
        "minset_ms_mean_ok": minset_ms_mean <= gates.max_minset_ms_mean,
        "eval_ms_mean_ok": eval_ms_mean <= gates.max_eval_ms_mean,
        "checks_total_ms_ok": checks_total_ms <= gates.max_checks_total_ms,
        "truncation_rate_ok": trunc_rate <= gates.max_truncation_rate,
        "baseline_spread_ok": certified_gap >= gates.min_oracle_minus_greedy_certified_gap,
    }
    failed_gates = [k for k, v in gate_results.items() if not v]

    return {
        "seed_base": seed_base,
        "counts": {"public": len(public_instances), "private": len(private_instances)},
        "timings_ms": {
            "generation_total": _ms(t0, t1),
            "generate_mean_per_instance": generate_ms_mean,
            "minset_mean_per_instance": minset_ms_mean,
            "evaluation_mean_per_agent_instance": eval_ms_mean,
            "priority_checks_total": checks_total_ms,
        },
        "search_stats": {
            "min_nodes": min(minset_nodes) if minset_nodes else 0,
            "max_nodes": max(minset_nodes) if minset_nodes else 0,
            "mean_nodes": _mean([float(x) for x in minset_nodes]),
            "truncation_rate": trunc_rate,
        },
        "calibration_stats": {
            "greedy_private_certified_rate": greedy_private.get("certified_rate", 0.0),
            "oracle_private_certified_rate": oracle_private.get("certified_rate", 0.0),
            "oracle_minus_greedy_certified_gap": certified_gap,
        },
        "priority_checks": checks,
        "gates": {
            "thresholds": asdict(gates),
            "results": gate_results,
            "failed": failed_gates,
            "passed_all": len(failed_gates) == 0,
        },
    }
