"""
Reference baseline agents for sanity-checking the GF-01 task.

These agents span a range from weak/random to strong/oracle-like search. They
are not the benchmark target; they are calibration tools to ensure the task is
solvable, discriminative, and consistent across evaluation tracks.
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

import random
from dataclasses import dataclass

from .models import GF01Instance, InterventionAtom
from .semantics import apply_certificate, run_automaton, sorted_certificate, timestep_cost
from .verifier import candidate_atoms, evaluate_certificate, find_valid_certificates


@dataclass
class BaselineAgent:
    name: str

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        raise NotImplementedError


class RandomInterventionAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__(name="BL-00-RandomIntervention")

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        rng = random.Random((instance.seed << 8) ^ seed)
        atoms = candidate_atoms(instance)
        rng.shuffle(atoms)
        cert: list[InterventionAtom] = []
        used_t: set[int] = set()
        for atom in atoms:
            if len(cert) >= instance.budget_atoms:
                break
            t_next = used_t | {atom.timestep}
            if len(t_next) > instance.budget_timestep:
                continue
            if rng.random() < 0.35:
                cert.append(atom)
                used_t.add(atom.timestep)
        return sorted_certificate(cert)


def _effect_hit_count(instance: GF01Instance, certificate: list[InterventionAtom]) -> int:
    trace = apply_certificate(instance.base_trace, certificate)
    _, outputs = run_automaton(instance.automaton, trace)
    if instance.mode == "hard":
        return int(outputs[instance.t_star].get(instance.effect_ap, 0))
    start = max(0, instance.t_star - instance.window_size)
    end = instance.t_star
    return sum(int(outputs[t].get(instance.effect_ap, 0)) for t in range(start, end + 1))


class GreedyLocalAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__(name="BL-01-GreedyLocal")

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        cert: list[InterventionAtom] = []
        remaining = candidate_atoms(instance)
        current_hits = _effect_hit_count(instance, cert)
        while len(cert) < instance.budget_atoms:
            best_atom = None
            best_score = (-1, -1, "")
            for atom in remaining:
                trial = sorted_certificate(cert + [atom])
                if timestep_cost(trial) > instance.budget_timestep:
                    continue
                res = evaluate_certificate(instance, trial)
                hits = _effect_hit_count(instance, trial)
                score = (
                    int(res.suff),  # prioritize making the effect happen
                    hits,           # then maximize local effect activations
                    f"{atom.timestep}:{atom.ap}:{atom.value}",
                )
                if score > best_score:
                    best_score = score
                    best_atom = atom
            if best_atom is None:
                break
            trial = sorted_certificate(cert + [best_atom])
            hits = _effect_hit_count(instance, trial)
            if hits < current_hits:
                break
            cert = trial
            current_hits = hits
            remaining = [a for a in remaining if a != best_atom]
            if evaluate_certificate(instance, cert).suff:
                break
        # Greedy prune to reduce obvious over-intervention while keeping sufficiency.
        pruned = cert[:]
        for atom in cert:
            trial = [a for a in pruned if a != atom]
            if evaluate_certificate(instance, trial).suff:
                pruned = trial
        return sorted_certificate(pruned)


class BudgetAwareSearchAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__(name="BL-02-BudgetAwareSearch")

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        search = find_valid_certificates(
            instance, max_nodes=min(120_000, instance.budget_atoms * 40_000), max_results=8
        )
        if not search.certificates:
            return []
        # Deterministic best certificate by lexicographic efficiency.
        return min(search.certificates, key=lambda c: (timestep_cost(c), len(c), [a.to_tuple() for a in c]))


class ToolPlannerAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__(name="BL-03-ToolPlanner")

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        # In this local harness, tool planner uses a stronger bounded exact search.
        search = find_valid_certificates(instance, max_nodes=250_000, max_results=16)
        if not search.certificates:
            return []
        return min(search.certificates, key=lambda c: (timestep_cost(c), len(c), [a.to_tuple() for a in c]))


class ExactOracleAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__(name="BL-04-ExactOracle")

    def propose(self, instance: GF01Instance, seed: int = 0) -> list[InterventionAtom]:
        search = find_valid_certificates(instance, max_nodes=600_000, max_results=32)
        if not search.certificates:
            return []
        return min(search.certificates, key=lambda c: (timestep_cost(c), len(c), [a.to_tuple() for a in c]))


def make_agent(agent_id: str) -> BaselineAgent:
    key = agent_id.strip().lower()
    if key in {"random", "bl-00", "bl-00-randomintervention"}:
        return RandomInterventionAgent()
    if key in {"greedy", "bl-01", "bl-01-greedylocal"}:
        return GreedyLocalAgent()
    if key in {"search", "bl-02", "bl-02-budgetawaresearch"}:
        return BudgetAwareSearchAgent()
    if key in {"tool", "bl-03", "bl-03-toolplanner"}:
        return ToolPlannerAgent()
    if key in {"oracle", "bl-04", "bl-04-exactoracle"}:
        return ExactOracleAgent()
    raise ValueError(f"unknown agent id: {agent_id}")
