"""
Deterministic generator for playable and verifiable GF-01 instances.

Starting from a seed, this module builds a complete automaton, a finite input
trace, and benchmark budgets/mode settings. It then rejects trivial or invalid
candidates until it finds an instance with an exact witness certificate.
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

import hashlib
import random

from .models import GF01Instance, GeneratorConfig, MealyAutomaton, Valuation
from .semantics import all_input_valuations, effect_satisfied, input_key, run_automaton
from .verifier import exact_existence_check


def _hash_id(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _generate_automaton(
    rng: random.Random, input_aps: list[str], output_aps: list[str], n_states: int
) -> MealyAutomaton:
    states = [f"s{i}" for i in range(n_states)]
    transitions: dict[str, dict[str, tuple[str, Valuation]]] = {}
    valuations = all_input_valuations(input_aps)
    for state in states:
        transitions[state] = {}
        for valuation in valuations:
            key = input_key(input_aps, valuation)
            next_state = states[rng.randrange(0, n_states)]
            output = {ap: rng.randrange(0, 2) for ap in output_aps}
            transitions[state][key] = (next_state, output)
    return MealyAutomaton(
        states=states,
        initial_state=states[0],
        input_aps=input_aps,
        output_aps=output_aps,
        transitions=transitions,
    )


def _generate_trace(rng: random.Random, t_len: int, input_aps: list[str]) -> list[Valuation]:
    trace: list[Valuation] = []
    for _ in range(t_len):
        trace.append({ap: rng.randrange(0, 2) for ap in input_aps})
    return trace


def _normalized_complexity(
    n_states: int, t_len: int, n_inputs: int, t_star: int
) -> dict[str, float]:
    # Placeholder normalization for initial harness. Calibrated tuning is handled post-pilot.
    state_norm = min(1.0, n_states / 10.0)
    depth_norm = min(1.0, (t_star + 1) / max(1, t_len))
    transitions_norm = min(1.0, (n_states * (2**n_inputs)) / 128.0)
    input_norm = min(1.0, n_inputs / 8.0)
    return {
        "state_count_norm": state_norm,
        "effect_depth_norm": depth_norm,
        "transition_count_norm": transitions_norm,
        "input_count_norm": input_norm,
    }


def _complexity_score(complexity: dict[str, float]) -> float:
    return sum(complexity.values()) / float(len(complexity))


def _window_size(cfg: GeneratorConfig, t_len: int, z_score: float) -> int:
    raw = round(cfg.alpha_0 + cfg.alpha_t * t_len + cfg.alpha_c * z_score)
    return max(cfg.w_min, min(cfg.w_max, raw))


def _budget_timestep(cfg: GeneratorConfig, t_len: int, z_score: float) -> int:
    # Slightly larger budgets for higher complexity in this first implementation.
    raw = round(max(1, cfg.budget_tightness * t_len + 0.25 * z_score * t_len))
    return max(1, min(t_len, raw))


def _budget_atoms(t_len: int, n_inputs: int, budget_timestep: int) -> int:
    # Unrestricted per-step semantics; only global budget applies.
    # Keep finite search tractable for the exact checker by defaulting to a moderate global cap.
    max_possible = t_len * n_inputs
    raw = max(1, budget_timestep + 1)
    return min(max_possible, raw)


def generate_instance(
    seed: int,
    cfg: GeneratorConfig | None = None,
    mode: str | None = None,
    split_id: str = "public_dev",
    input_aps: list[str] | None = None,
    output_aps: list[str] | None = None,
    trace_len: int | None = None,
) -> tuple[GF01Instance, list]:
    cfg = cfg or GeneratorConfig()
    rng = random.Random(seed)
    input_aps = input_aps or ["in0", "in1", "in2"]
    output_aps = output_aps or ["out0", "out1"]

    for attempt in range(cfg.max_generation_attempts):
        t_len = trace_len if trace_len is not None else rng.randint(cfg.trace_min, cfg.trace_max)
        n_states = rng.randint(cfg.states_min, cfg.states_max)
        automaton = _generate_automaton(rng, input_aps, output_aps, n_states)
        base_trace = _generate_trace(rng, t_len, input_aps)
        resolved_mode = mode or ("hard" if rng.random() < 0.35 else "normal")
        t_star = rng.randint(max(1, t_len // 3), t_len - 1)
        effect_ap = output_aps[rng.randrange(0, len(output_aps))]
        complexity = _normalized_complexity(n_states, t_len, len(input_aps), t_star)
        z_score = _complexity_score(complexity)
        w_size = _window_size(cfg, t_len, z_score) if resolved_mode == "normal" else 1
        b_t = _budget_timestep(cfg, t_len, z_score)
        b_a = _budget_atoms(t_len, len(input_aps), b_t)

        raw_id = f"{seed}:{attempt}:{resolved_mode}:{t_len}:{n_states}:{effect_ap}:{t_star}:{split_id}"
        instance = GF01Instance(
            instance_id=f"gf01-{_hash_id(raw_id)}",
            automaton=automaton,
            base_trace=base_trace,
            effect_ap=effect_ap,
            t_star=t_star,
            mode=resolved_mode,
            window_size=w_size,
            budget_timestep=b_t,
            budget_atoms=b_a,
            seed=seed,
            complexity=complexity,
            split_id=split_id,
            renderer_track="json",
        )
        # Reject trivial instances solved by empty certificate.
        _, base_outputs = run_automaton(instance.automaton, instance.base_trace)
        if effect_satisfied(instance, base_outputs):
            continue
        exists, witness, truncated = exact_existence_check(
            instance, max_nodes=cfg.max_exact_nodes
        )
        if exists and witness is not None:
            return instance, witness
        # If search truncates, retry to avoid accepting unknown/no-witness instances.
        if truncated:
            continue
    raise RuntimeError("failed to generate an accepted instance within max attempts")


def generate_suite(
    seeds: list[int],
    split_id: str,
    cfg: GeneratorConfig | None = None,
    mode: str | None = None,
) -> list[GF01Instance]:
    cfg = cfg or GeneratorConfig()
    suite: list[GF01Instance] = []
    for seed in seeds:
        instance, _ = generate_instance(seed=seed, cfg=cfg, split_id=split_id, mode=mode)
        suite.append(instance)
    return suite
