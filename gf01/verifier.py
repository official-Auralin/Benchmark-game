"""
Exact certificate checking for sufficiency, minimality, and validity.

Given a proposed intervention set, this module enforces structure rules,
executes the counterfactual rollout, and confirms whether the proposal is both
effective and minimally necessary under the benchmark definition.
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

from dataclasses import dataclass

from .models import GF01Instance, InterventionAtom
from .semantics import (
    apply_certificate,
    atom_cost,
    effect_satisfied,
    run_automaton,
    sorted_certificate,
    timestep_cost,
)


@dataclass
class VerificationResult:
    suff: bool
    min1: bool
    valid: bool
    goal: bool


@dataclass
class SearchResult:
    certificates: list[list[InterventionAtom]]
    nodes_visited: int
    truncated: bool


def validate_certificate_structure(
    instance: GF01Instance, certificate: list[InterventionAtom]
) -> tuple[bool, str]:
    seen: dict[tuple[int, str], int] = {}
    max_t = len(instance.base_trace) - 1
    for atom in certificate:
        if atom.timestep < 0 or atom.timestep > max_t:
            return False, f"timestep {atom.timestep} out of range"
        if atom.ap not in instance.automaton.input_aps:
            return False, f"ap {atom.ap} not in AP_in"
        if atom.value not in (0, 1):
            return False, f"value {atom.value} is invalid"
        key = (atom.timestep, atom.ap)
        if key in seen and seen[key] != atom.value:
            return False, f"conflicting assignments for {key}"
        seen[key] = atom.value
    if timestep_cost(certificate) > instance.budget_timestep:
        return False, "timestep budget exceeded"
    if atom_cost(certificate) > instance.budget_atoms:
        return False, "atom budget exceeded"
    return True, ""


def evaluate_certificate(
    instance: GF01Instance, certificate: list[InterventionAtom]
) -> VerificationResult:
    ok, _ = validate_certificate_structure(instance, certificate)
    if not ok:
        return VerificationResult(False, False, False, False)

    intervened_trace = apply_certificate(instance.base_trace, certificate)
    _, outputs = run_automaton(instance.automaton, intervened_trace)
    suff = effect_satisfied(instance, outputs)
    if not suff:
        return VerificationResult(False, False, False, False)

    min1 = True
    for idx in range(len(certificate)):
        reduced = certificate[:idx] + certificate[idx + 1 :]
        reduced_trace = apply_certificate(instance.base_trace, reduced)
        _, reduced_outputs = run_automaton(instance.automaton, reduced_trace)
        if effect_satisfied(instance, reduced_outputs):
            min1 = False
            break
    valid = suff and min1
    return VerificationResult(suff, min1, valid, suff)


def candidate_atoms(instance: GF01Instance) -> list[InterventionAtom]:
    atoms: list[InterventionAtom] = []
    for t, step in enumerate(instance.base_trace):
        for ap in instance.automaton.input_aps:
            base_v = int(step[ap])
            for v in (0, 1):
                if v != base_v:
                    atoms.append(InterventionAtom(timestep=t, ap=ap, value=v))
    return atoms


def find_valid_certificates(
    instance: GF01Instance, max_nodes: int, max_results: int | None = None
) -> SearchResult:
    atoms = candidate_atoms(instance)
    found: list[list[InterventionAtom]] = []
    nodes = 0

    # Include empty certificate for completeness.
    empty_res = evaluate_certificate(instance, [])
    nodes += 1
    if empty_res.valid:
        found.append([])

    if max_results is not None and len(found) >= max_results:
        return SearchResult(found, nodes, False)

    max_atoms_given_timestep_budget = int(
        instance.budget_timestep * len(instance.automaton.input_aps)
    )

    truncated = False

    def _search_k(
        *,
        target_k: int,
        start_idx: int,
        chosen: list[InterventionAtom],
        used_timesteps: set[int],
    ) -> None:
        nonlocal nodes, truncated
        if truncated:
            return
        if max_results is not None and len(found) >= max_results:
            return

        if len(chosen) == target_k:
            nodes += 1
            if nodes > max_nodes:
                truncated = True
                return
            cert = sorted_certificate(chosen)
            res = evaluate_certificate(instance, cert)
            if res.valid:
                found.append(cert)
            return

        remaining = target_k - len(chosen)
        upper = len(atoms) - remaining
        for idx in range(start_idx, upper + 1):
            atom = atoms[idx]
            timestep_added = atom.timestep not in used_timesteps
            if timestep_added and (len(used_timesteps) + 1) > instance.budget_timestep:
                continue

            chosen.append(atom)
            if timestep_added:
                used_timesteps.add(atom.timestep)
            _search_k(
                target_k=target_k,
                start_idx=idx + 1,
                chosen=chosen,
                used_timesteps=used_timesteps,
            )
            if timestep_added:
                used_timesteps.remove(atom.timestep)
            chosen.pop()

            if truncated:
                return
            if max_results is not None and len(found) >= max_results:
                return

    for k in range(1, instance.budget_atoms + 1):
        if k > max_atoms_given_timestep_budget:
            break
        _search_k(
            target_k=k,
            start_idx=0,
            chosen=[],
            used_timesteps=set(),
        )
        if truncated:
            return SearchResult(found, nodes, True)
        if max_results is not None and len(found) >= max_results:
            return SearchResult(found, nodes, False)
    return SearchResult(found, nodes, False)


def exact_existence_check(instance: GF01Instance, max_nodes: int) -> tuple[bool, list[InterventionAtom] | None, bool]:
    result = find_valid_certificates(instance, max_nodes=max_nodes, max_results=1)
    if result.certificates:
        return True, result.certificates[0], result.truncated
    return False, None, result.truncated
