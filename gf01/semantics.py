"""
Core GF-01 mechanics for traces, interventions, and observations.

This module is the "game rules engine." It applies chosen input changes to a
trace, runs the automaton step by step, checks whether the target effect was
achieved, and handles observation rendering/parsing for parity checks.
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

import itertools
import json
from collections.abc import Iterable

from .models import GF01Instance, InterventionAtom, MealyAutomaton, Valuation


def input_key(input_aps: list[str], valuation: Valuation) -> str:
    return "|".join(f"{ap}={int(valuation.get(ap, 0))}" for ap in input_aps)


def all_input_valuations(input_aps: list[str]) -> list[Valuation]:
    values: list[Valuation] = []
    for bits in itertools.product((0, 1), repeat=len(input_aps)):
        values.append({ap: int(b) for ap, b in zip(input_aps, bits)})
    return values


def sorted_certificate(certificate: Iterable[InterventionAtom]) -> list[InterventionAtom]:
    return sorted(certificate, key=lambda c: (c.timestep, c.ap, c.value))


def apply_certificate(
    base_trace: list[Valuation], certificate: Iterable[InterventionAtom]
) -> list[Valuation]:
    trace = [{k: int(v) for k, v in step.items()} for step in base_trace]
    # Deterministic overwrite map for set semantics.
    edits: dict[tuple[int, str], int] = {}
    for atom in sorted_certificate(certificate):
        edits[(atom.timestep, atom.ap)] = int(atom.value)
    for (t, ap), v in edits.items():
        trace[t][ap] = v
    return trace


def run_automaton(
    automaton: MealyAutomaton, input_trace: list[Valuation]
) -> tuple[list[str], list[Valuation]]:
    state = automaton.initial_state
    state_trace = [state]
    outputs: list[Valuation] = []
    for step in input_trace:
        key = input_key(automaton.input_aps, step)
        if state not in automaton.transitions or key not in automaton.transitions[state]:
            raise ValueError(f"missing transition for state={state} key={key}")
        next_state, out = automaton.transitions[state][key]
        outputs.append({k: int(out.get(k, 0)) for k in automaton.output_aps})
        state = next_state
        state_trace.append(state)
    return state_trace, outputs


def effect_hard(outputs: list[Valuation], effect_ap: str, t_star: int) -> bool:
    return bool(outputs[t_star].get(effect_ap, 0))


def effect_normal(
    outputs: list[Valuation], effect_ap: str, t_star: int, window_size: int
) -> bool:
    start = max(0, t_star - window_size)
    end = min(len(outputs) - 1, t_star)
    for t in range(start, end + 1):
        if outputs[t].get(effect_ap, 0) == 1:
            return True
    return False


def effect_satisfied(instance: GF01Instance, outputs: list[Valuation]) -> bool:
    if instance.mode == "hard":
        return effect_hard(outputs, instance.effect_ap, instance.t_star)
    return effect_normal(outputs, instance.effect_ap, instance.t_star, instance.window_size)


def timestep_cost(certificate: Iterable[InterventionAtom]) -> int:
    return len({atom.timestep for atom in certificate})


def atom_cost(certificate: Iterable[InterventionAtom]) -> int:
    return len(list(certificate))


def canonical_observation(
    t: int,
    outputs: list[Valuation],
    effect_triggered: bool,
    budget_t_remaining: int,
    budget_a_remaining: int,
    history: list[InterventionAtom],
    mode: str,
    t_star: int,
) -> dict[str, object]:
    return {
        "t": int(t),
        "y_t": {k: int(v) for k, v in sorted(outputs[t].items())},
        "effect_status_t": "triggered" if effect_triggered else "not-triggered",
        "budget_t_remaining": int(budget_t_remaining),
        "budget_a_remaining": int(budget_a_remaining),
        "history_atoms": [atom.to_tuple() for atom in sorted_certificate(history)],
        "mode": mode,
        "t_star": int(t_star),
    }


def render_json(obs: dict[str, object]) -> str:
    return json.dumps(obs, sort_keys=True, separators=(",", ":"))


def parse_json(rendered: str) -> dict[str, object]:
    return json.loads(rendered)


def render_visual(obs: dict[str, object]) -> str:
    # Human-facing view with an embedded canonical payload for exact parsing.
    t_now = int(obs["t"])
    t_star = int(obs["t_star"])
    mode = str(obs["mode"])
    effect_status = str(obs["effect_status_t"])
    budget_t = int(obs["budget_t_remaining"])
    budget_a = int(obs["budget_a_remaining"])
    y_t = obs.get("y_t", {})
    history_atoms = obs.get("history_atoms", [])

    if isinstance(y_t, dict):
        y_tokens = [f"{k}={int(v)}" for k, v in sorted(y_t.items())]
        y_text = " ".join(y_tokens) if y_tokens else "(none)"
    else:
        y_text = "(invalid)"

    history_lines = ["Interventions so far:"]
    grouped: dict[int, list[tuple[str, int]]] = {}
    if isinstance(history_atoms, list):
        for item in history_atoms:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            t_item = int(item[0])
            grouped.setdefault(t_item, []).append((str(item[1]), int(item[2])))
    if grouped:
        for t_item in sorted(grouped):
            parts = [f"{ap}={value}" for ap, value in sorted(grouped[t_item])]
            history_lines.append(f"  t={t_item}: " + ", ".join(parts))
    else:
        history_lines.append("  (none)")

    lines = [
        "=== GF-01 Visual Snapshot ===",
        f"Time: t={t_now} (target t*={t_star}, mode={mode})",
        f"Effect status: {effect_status}",
        f"Budget remaining: timesteps={budget_t}, atoms={budget_a}",
        f"Outputs y_t: {y_text}",
        *history_lines,
        # Exact parse anchor used by parse_visual for parity checks.
        f"OBS_JSON={render_json(obs)}",
    ]
    return "\n".join(lines)


def parse_visual(rendered: str) -> dict[str, object]:
    lines = rendered.splitlines()
    for line in reversed(lines):
        if line.startswith("OBS_JSON="):
            payload = line.split("=", 1)[1]
            return json.loads(payload)

    # Backward compatibility for legacy visual renderer snapshots.
    kv: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        kv[key] = value
    required = {"T", "MODE", "TSTAR", "EFFECT", "YT", "BT", "BA", "H"}
    if not required.issubset(kv):
        raise ValueError("unsupported visual rendering format")
    return {
        "t": int(kv["T"]),
        "mode": kv["MODE"],
        "t_star": int(kv["TSTAR"]),
        "effect_status_t": kv["EFFECT"],
        "y_t": json.loads(kv["YT"]),
        "budget_t_remaining": int(kv["BT"]),
        "budget_a_remaining": int(kv["BA"]),
        "history_atoms": json.loads(kv["H"]),
    }
