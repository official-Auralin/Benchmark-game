"""
Data models for GF-01 instances, interventions, automata, and run outputs.

This module defines the core typed objects used across the harness. These
classes keep the benchmark state explicit and serializable, so generation,
verification, evaluation, and reporting all operate on the same data shape.
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

from dataclasses import dataclass, field
from typing import Any


Bit = int
Valuation = dict[str, Bit]


@dataclass(frozen=True, order=True)
class InterventionAtom:
    timestep: int
    ap: str
    value: Bit

    def to_tuple(self) -> tuple[int, str, Bit]:
        return (self.timestep, self.ap, self.value)


@dataclass
class MealyAutomaton:
    states: list[str]
    initial_state: str
    input_aps: list[str]
    output_aps: list[str]
    # transitions[state][input_key] = (next_state, output_valuation)
    transitions: dict[str, dict[str, tuple[str, Valuation]]]

    def to_canonical_dict(self) -> dict[str, Any]:
        canon: dict[str, Any] = {
            "states": sorted(self.states),
            "initial_state": self.initial_state,
            "input_aps": list(self.input_aps),
            "output_aps": list(self.output_aps),
            "transitions": {},
        }
        for state in sorted(self.transitions):
            canon["transitions"][state] = {}
            for key in sorted(self.transitions[state]):
                nxt, out = self.transitions[state][key]
                canon["transitions"][state][key] = {
                    "next_state": nxt,
                    "output": {k: int(out[k]) for k in sorted(out)},
                }
        return canon


@dataclass
class GF01Instance:
    instance_id: str
    automaton: MealyAutomaton
    base_trace: list[Valuation]
    effect_ap: str
    t_star: int
    mode: str  # "normal" | "hard"
    window_size: int
    budget_timestep: int
    budget_atoms: int
    seed: int
    complexity: dict[str, float] = field(default_factory=dict)
    split_id: str = "public_dev"
    renderer_track: str = "json"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "automaton": self.automaton.to_canonical_dict(),
            "base_trace": [
                {k: int(step[k]) for k in sorted(step)} for step in self.base_trace
            ],
            "effect_ap": self.effect_ap,
            "t_star": int(self.t_star),
            "mode": self.mode,
            "window_size": int(self.window_size),
            "budget_timestep": int(self.budget_timestep),
            "budget_atoms": int(self.budget_atoms),
            "seed": int(self.seed),
            "complexity": {k: float(self.complexity[k]) for k in sorted(self.complexity)},
            "split_id": self.split_id,
            "renderer_track": self.renderer_track,
        }


@dataclass
class GeneratorConfig:
    states_min: int = 3
    states_max: int = 6
    trace_min: int = 6
    trace_max: int = 12
    alpha_0: float = 1.0
    alpha_t: float = 0.0
    alpha_c: float = 2.0
    w_min: int = 1
    w_max: int = 6
    budget_tightness: float = 0.25
    max_generation_attempts: int = 20
    max_exact_nodes: int = 120_000


@dataclass
class RunRecord:
    instance_id: str
    eval_track: str
    renderer_track: str
    agent_name: str
    certificate: list[InterventionAtom]
    suff: bool
    min1: bool
    valid: bool
    goal: bool
    eff_t: int
    eff_a: int
    ap_precision: float
    ap_recall: float
    ap_f1: float
    ts_precision: float
    ts_recall: float
    ts_f1: float
    diagnostic_status: str = "not-run"
    diagnostic_runtime_ms: int = 0
