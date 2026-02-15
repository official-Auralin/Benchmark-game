"""
Harness checks for correctness, parity, determinism, and shortcut resistance.

This module runs test-style validations that protect benchmark integrity. It
checks renderer parity, deterministic replay, certificate-structure edge cases,
and behavior patterns that indicate superficial shortcutting.
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

from .baselines import (
    BaselineAgent,
    BudgetAwareSearchAgent,
    ExactOracleAgent,
    GreedyLocalAgent,
    RandomInterventionAgent,
)
from .generator import generate_instance, generate_suite
from .metrics import aggregate_suite, compute_ap_ts_metrics
from .models import GF01Instance, GeneratorConfig, InterventionAtom, RunRecord
from .semantics import (
    canonical_observation,
    parse_json,
    parse_visual,
    render_json,
    render_visual,
    sorted_certificate,
    timestep_cost,
)
from .verifier import evaluate_certificate, find_valid_certificates, validate_certificate_structure


def _minimal_set(instance: GF01Instance, max_nodes: int = 500_000) -> list[list]:
    return find_valid_certificates(instance, max_nodes=max_nodes, max_results=32).certificates


def evaluate_agent_on_instance(
    agent: BaselineAgent,
    instance: GF01Instance,
    eval_track: str,
    renderer_track: str,
    seed: int = 0,
    minimal_set: list[list] | None = None,
) -> RunRecord:
    if minimal_set is None:
        minimal_set = _minimal_set(instance, max_nodes=120_000)
    cert = sorted_certificate(agent.propose(instance, seed=seed))
    ver = evaluate_certificate(instance, cert)
    ap_p, ap_r, ap_f1, ts_p, ts_r, ts_f1 = compute_ap_ts_metrics(cert, minimal_set)
    return RunRecord(
        instance_id=instance.instance_id,
        eval_track=eval_track,
        renderer_track=renderer_track,
        agent_name=agent.name,
        certificate=cert,
        suff=ver.suff,
        min1=ver.min1,
        valid=ver.valid,
        goal=ver.goal,
        eff_t=timestep_cost(cert),
        eff_a=len(cert),
        ap_precision=ap_p,
        ap_recall=ap_r,
        ap_f1=ap_f1,
        ts_precision=ts_p,
        ts_recall=ts_r,
        ts_f1=ts_f1,
    )


def evaluate_suite(
    agent: BaselineAgent,
    instances: list[GF01Instance],
    eval_track: str,
    renderer_track: str,
    seed: int = 0,
    minimal_cache: dict[str, list[list]] | None = None,
) -> tuple[list[RunRecord], dict[str, float]]:
    if minimal_cache is None:
        minimal_cache = {
            instance.instance_id: _minimal_set(instance, max_nodes=120_000) for instance in instances
        }
    records = []
    for instance in instances:
        records.append(
            evaluate_agent_on_instance(
                agent=agent,
                instance=instance,
                eval_track=eval_track,
                renderer_track=renderer_track,
                seed=seed,
                minimal_set=minimal_cache[instance.instance_id],
            )
        )
    return records, aggregate_suite(records)


def check_ab001_renderer_parity(
    instances: list[GF01Instance], tolerance: float = 0.0
) -> dict[str, object]:
    roundtrip_failures = 0
    for instance in instances:
        # Build canonical observations on zero-history baseline rollout.
        cert = []
        # Build observations from static budgets and empty history at each t.
        # This checks renderer mapping parity, not policy quality.
        from .semantics import apply_certificate, run_automaton, effect_satisfied

        trace = apply_certificate(instance.base_trace, cert)
        _, outputs = run_automaton(instance.automaton, trace)
        for t in range(len(outputs)):
            obs = canonical_observation(
                t=t,
                outputs=outputs,
                effect_triggered=effect_satisfied(instance, outputs),
                budget_t_remaining=instance.budget_timestep,
                budget_a_remaining=instance.budget_atoms,
                history=[],
                mode=instance.mode,
                t_star=instance.t_star,
            )
            obs_json = parse_json(render_json(obs))
            obs_visual = parse_visual(render_visual(obs))
            if json.dumps(obs_json, sort_keys=True) != json.dumps(obs_visual, sort_keys=True):
                roundtrip_failures += 1

    agent = RandomInterventionAgent()
    _, agg_json = evaluate_suite(agent, instances, eval_track="EVAL-CB", renderer_track="json", seed=123)
    _, agg_visual = evaluate_suite(agent, instances, eval_track="EVAL-CB", renderer_track="visual", seed=123)
    certified_gap = abs(agg_json.get("certified_rate", 0.0) - agg_visual.get("certified_rate", 0.0))
    passed = roundtrip_failures == 0 and certified_gap <= tolerance
    return {
        "check_id": "AB-001",
        "passed": passed,
        "roundtrip_failures": roundtrip_failures,
        "certified_rate_gap": certified_gap,
        "tolerance": tolerance,
    }


def check_ab003_greedy_vs_certified(
    private_instances: list[GF01Instance],
    goal_threshold: float = 0.40,
    certified_floor: float = 0.05,
) -> dict[str, object]:
    greedy = GreedyLocalAgent()
    oracle = ExactOracleAgent()
    _, g_agg = evaluate_suite(greedy, private_instances, eval_track="EVAL-CB", renderer_track="json", seed=10)
    _, o_agg = evaluate_suite(oracle, private_instances, eval_track="EVAL-OC", renderer_track="json", seed=10)

    greedy_goal = g_agg.get("goal_rate", 0.0)
    greedy_cert = g_agg.get("certified_rate", 0.0)
    shortcut_flag = greedy_goal > goal_threshold and greedy_cert < certified_floor
    passed = not shortcut_flag
    return {
        "check_id": "AB-003",
        "passed": passed,
        "greedy_goal_rate": greedy_goal,
        "greedy_certified_rate": greedy_cert,
        "oracle_certified_rate": o_agg.get("certified_rate", 0.0),
        "goal_threshold": goal_threshold,
        "certified_floor": certified_floor,
    }


def check_ab005_public_private_gap(
    public_instances: list[GF01Instance],
    private_instances: list[GF01Instance],
    gap_bound: float = 0.15,
) -> dict[str, object]:
    agent = BudgetAwareSearchAgent()
    _, pub = evaluate_suite(agent, public_instances, eval_track="EVAL-CB", renderer_track="json", seed=20)
    _, prv = evaluate_suite(agent, private_instances, eval_track="EVAL-CB", renderer_track="json", seed=20)
    gap = abs(pub.get("certified_rate", 0.0) - prv.get("certified_rate", 0.0))
    return {
        "check_id": "AB-005",
        "passed": gap <= gap_bound,
        "public_certified_rate": pub.get("certified_rate", 0.0),
        "private_certified_rate": prv.get("certified_rate", 0.0),
        "gap": gap,
        "gap_bound": gap_bound,
    }


def check_ab007_deterministic_replay(
    seed: int = 9991, cfg: GeneratorConfig | None = None
) -> dict[str, object]:
    cfg = cfg or GeneratorConfig()
    i1, w1 = generate_instance(seed=seed, cfg=cfg, split_id="public_dev")
    i2, w2 = generate_instance(seed=seed, cfg=cfg, split_id="public_dev")
    same_instance = json.dumps(i1.to_canonical_dict(), sort_keys=True) == json.dumps(
        i2.to_canonical_dict(), sort_keys=True
    )
    same_witness = [a.to_tuple() for a in w1] == [a.to_tuple() for a in w2]
    return {
        "check_id": "AB-007",
        "passed": same_instance and same_witness,
        "same_instance": same_instance,
        "same_witness": same_witness,
    }


def _cert_exceeding_timestep_budget(instance: GF01Instance) -> list[InterventionAtom] | None:
    needed = instance.budget_timestep + 1
    if needed > len(instance.base_trace):
        return None
    ap = instance.automaton.input_aps[0]
    cert: list[InterventionAtom] = []
    for t in range(needed):
        v = 1 - int(instance.base_trace[t][ap])
        cert.append(InterventionAtom(timestep=t, ap=ap, value=v))
    return cert


def _cert_exceeding_atom_budget(instance: GF01Instance) -> list[InterventionAtom]:
    needed = instance.budget_atoms + 1
    cert: list[InterventionAtom] = []
    for t in range(len(instance.base_trace)):
        for ap in instance.automaton.input_aps:
            v = 1 - int(instance.base_trace[t][ap])
            cert.append(InterventionAtom(timestep=t, ap=ap, value=v))
            if len(cert) >= needed:
                return cert
    # Fallback: duplicate first atom to force atom-count overflow.
    if not cert:
        cert = [InterventionAtom(timestep=0, ap=instance.automaton.input_aps[0], value=1)]
    while len(cert) < needed:
        cert.append(cert[0])
    return cert


def check_ab009_certificate_structure_edges(
    seed: int = 9101, cfg: GeneratorConfig | None = None
) -> dict[str, object]:
    cfg = cfg or GeneratorConfig()
    instance, _ = generate_instance(seed=seed, cfg=cfg, split_id="public_dev")
    ap = instance.automaton.input_aps[0]
    flip = 1 - int(instance.base_trace[0][ap])

    cases: dict[str, tuple[list[InterventionAtom], str]] = {
        "neg_timestep": ([InterventionAtom(-1, ap, flip)], "out of range"),
        "high_timestep": ([InterventionAtom(len(instance.base_trace), ap, flip)], "out of range"),
        "non_input_ap": ([InterventionAtom(0, "__bad_input_ap__", flip)], "not in AP_in"),
        "invalid_value": ([InterventionAtom(0, ap, 2)], "invalid"),
        "conflicting_assignments": (
            [InterventionAtom(0, ap, 0), InterventionAtom(0, ap, 1)],
            "conflicting assignments",
        ),
        "atom_budget_exceeded": (
            _cert_exceeding_atom_budget(instance),
            "atom budget exceeded",
        ),
    }
    over_timestep = _cert_exceeding_timestep_budget(instance)
    if over_timestep is not None:
        cases["timestep_budget_exceeded"] = (over_timestep, "timestep budget exceeded")

    per_case: dict[str, object] = {}
    passed = True
    for name, (cert, expected_reason) in cases.items():
        ok, reason = validate_certificate_structure(instance, cert)
        ver = evaluate_certificate(instance, cert)
        case_pass = (not ok) and (expected_reason in reason) and (not ver.valid)
        passed = passed and case_pass
        per_case[name] = {
            "passed": case_pass,
            "reason": reason,
            "expected_reason": expected_reason,
            "valid": ver.valid,
            "suff": ver.suff,
            "min1": ver.min1,
        }

    return {
        "check_id": "AB-009",
        "passed": passed,
        "cases": per_case,
    }


def run_priority_h3_checks(seed_base: int = 1000) -> dict[str, object]:
    cfg = GeneratorConfig()
    public_instances = generate_suite([seed_base + i for i in range(2)], split_id="public_dev", cfg=cfg)
    private_instances = generate_suite([seed_base + 100 + i for i in range(2)], split_id="private_eval", cfg=cfg)
    return {
        "AB-001": check_ab001_renderer_parity(public_instances),
        "AB-003": check_ab003_greedy_vs_certified(private_instances),
        "AB-005": check_ab005_public_private_gap(public_instances, private_instances),
        "AB-007": check_ab007_deterministic_replay(seed=seed_base + 777, cfg=cfg),
        "AB-009": check_ab009_certificate_structure_edges(seed=seed_base + 888, cfg=cfg),
    }
