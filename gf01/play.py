"""
Playable GF-01 episode loop with a stable observation/action contract.

This module adds the first human-playable and agent-playable runtime path for
GF-01. It executes one timestep at a time, exposes canonical observations
(`O(s)`), accepts per-timestep intervention actions, and returns a structured
machine-checkable episode artifact.
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines import BaselineAgent
from .models import GF01Instance, InterventionAtom, Valuation
from .semantics import (
    apply_certificate,
    atom_cost,
    canonical_observation,
    effect_satisfied,
    input_key,
    parse_json,
    parse_visual,
    render_json,
    render_visual,
    run_automaton,
    sorted_certificate,
    timestep_cost,
)
from .verifier import evaluate_certificate


PolicyFn = Callable[[dict[str, object] | None, int, GF01Instance], dict[str, int]]


def _remaining_budgets(
    instance: GF01Instance, history: list[InterventionAtom]
) -> tuple[int, int]:
    budget_t_remaining = instance.budget_timestep - timestep_cost(history)
    budget_a_remaining = instance.budget_atoms - atom_cost(history)
    return budget_t_remaining, budget_a_remaining


def _effect_triggered_prefix(instance: GF01Instance, outputs: list[Valuation]) -> bool:
    if not outputs:
        return False
    if instance.mode == "hard":
        if len(outputs) <= instance.t_star:
            return False
        return bool(outputs[instance.t_star].get(instance.effect_ap, 0))
    start = max(0, instance.t_star - instance.window_size)
    end = min(instance.t_star, len(outputs) - 1)
    if end < start:
        return False
    for t in range(start, end + 1):
        if int(outputs[t].get(instance.effect_ap, 0)) == 1:
            return True
    return False


def _coerce_bit(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"bit must be 0/1, got {value}")
    if isinstance(value, str):
        text = value.strip()
        if text in {"0", "1"}:
            return int(text)
    raise ValueError(f"cannot coerce to bit: {value!r}")


def normalize_step_action(
    instance: GF01Instance,
    timestep: int,
    raw_action: dict[str, Any],
) -> list[InterventionAtom]:
    if not isinstance(raw_action, dict):
        raise ValueError("step action must be an object mapping AP->bit")
    atoms: list[InterventionAtom] = []
    for ap, raw_value in sorted(raw_action.items()):
        if ap not in instance.automaton.input_aps:
            raise ValueError(f"unknown input AP: {ap}")
        value = _coerce_bit(raw_value)
        base_value = int(instance.base_trace[timestep][ap])
        if value == base_value:
            # No-op assignments are ignored to keep certificate semantics clean.
            continue
        atoms.append(InterventionAtom(timestep=timestep, ap=ap, value=value))
    return atoms


def _validate_step_budget(
    instance: GF01Instance,
    history: list[InterventionAtom],
    step_atoms: list[InterventionAtom],
) -> None:
    if not step_atoms:
        return
    used_timesteps = {atom.timestep for atom in history}
    used_timesteps.add(step_atoms[0].timestep)
    if len(used_timesteps) > instance.budget_timestep:
        raise ValueError("timestep budget exceeded")
    if len(history) + len(step_atoms) > instance.budget_atoms:
        raise ValueError("atom budget exceeded")


def _step_input(
    instance: GF01Instance, timestep: int, step_atoms: list[InterventionAtom]
) -> Valuation:
    step = {ap: int(v) for ap, v in instance.base_trace[timestep].items()}
    for atom in step_atoms:
        step[atom.ap] = int(atom.value)
    return step


def _render_observation(obs: dict[str, object], renderer_track: str) -> str:
    if renderer_track == "visual":
        return render_visual(obs)
    if renderer_track == "json":
        return render_json(obs)
    raise ValueError(f"unsupported renderer_track: {renderer_track}")


def parse_action_script(path: str) -> dict[int, dict[str, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_timestep: dict[int, dict[str, int]] = {}

    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("action rows must be objects")
            timestep = int(row.get("t"))
            edits = row.get("set", {})
            if not isinstance(edits, dict):
                raise ValueError("action row 'set' must be an object")
            by_timestep[timestep] = {str(ap): _coerce_bit(v) for ap, v in edits.items()}
        return by_timestep

    if isinstance(payload, dict):
        for key, edits in payload.items():
            timestep = int(key)
            if not isinstance(edits, dict):
                raise ValueError("dict-form action entries must map to objects")
            by_timestep[timestep] = {str(ap): _coerce_bit(v) for ap, v in edits.items()}
        return by_timestep

    raise ValueError("unsupported action script format")


def baseline_policy(agent: BaselineAgent, instance: GF01Instance) -> PolicyFn:
    full_certificate = sorted_certificate(agent.propose(instance, seed=instance.seed))
    actions_by_t: dict[int, dict[str, int]] = {}
    for atom in full_certificate:
        actions_by_t.setdefault(atom.timestep, {})
        actions_by_t[atom.timestep][atom.ap] = int(atom.value)

    def _policy(
        _last_obs: dict[str, object] | None, timestep: int, _instance: GF01Instance
    ) -> dict[str, int]:
        return dict(actions_by_t.get(timestep, {}))

    return _policy


def scripted_policy(actions_by_t: dict[int, dict[str, int]]) -> PolicyFn:
    def _policy(
        _last_obs: dict[str, object] | None, timestep: int, _instance: GF01Instance
    ) -> dict[str, int]:
        return dict(actions_by_t.get(timestep, {}))

    return _policy


def _parse_human_action(raw: str) -> dict[str, int]:
    text = raw.strip()
    if not text or text.lower() in {"skip", "none", "pass"}:
        return {}
    edits: dict[str, int] = {}
    for token in text.split(","):
        part = token.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid token (expected ap=value): {part}")
        ap, value_text = part.split("=", 1)
        ap = ap.strip()
        edits[ap] = _coerce_bit(value_text.strip())
    return edits


def human_policy(renderer_track: str = "visual") -> PolicyFn:
    def _policy(
        last_obs: dict[str, object] | None, timestep: int, instance: GF01Instance
    ) -> dict[str, int]:
        while True:
            print(f"\n--- Timestep {timestep} ---")
            if last_obs is None:
                print("No observation yet (start of episode).")
            else:
                rendered = _render_observation(last_obs, renderer_track)
                print("Last observation:")
                print(rendered)
                if renderer_track == "json":
                    # Contract sanity check: parse(render(obs)) round-trip.
                    _ = parse_json(rendered)
                else:
                    _ = parse_visual(rendered)
            budget_t_remaining = int(last_obs["budget_t_remaining"]) if last_obs else int(
                instance.budget_timestep
            )
            budget_a_remaining = int(last_obs["budget_a_remaining"]) if last_obs else int(
                instance.budget_atoms
            )
            print(
                f"Budget remaining: timestep={budget_t_remaining}, atoms={budget_a_remaining}"
            )
            print(
                "Enter action for THIS timestep only.\n"
                "  Accepted: 'skip' | 'in0=1' | 'in0=1,in2=0'\n"
                "  Values must be 0 or 1; AP names must match the valid list.\n"
                f"  Valid APs: {', '.join(instance.automaton.input_aps)}"
            )
            raw = input("> ")
            try:
                return _parse_human_action(raw)
            except ValueError as exc:
                print(f"Invalid action: {exc}")

    return _policy


def run_episode(
    instance: GF01Instance,
    policy: PolicyFn,
    *,
    renderer_track: str = "json",
) -> dict[str, object]:
    state = instance.automaton.initial_state
    outputs: list[Valuation] = []
    history: list[InterventionAtom] = []
    steps: list[dict[str, object]] = []
    last_observation: dict[str, object] | None = None

    for timestep in range(len(instance.base_trace)):
        raw_action = policy(last_observation, timestep, instance) or {}
        step_atoms = normalize_step_action(instance, timestep, raw_action)
        _validate_step_budget(instance, history, step_atoms)
        step_input = _step_input(instance, timestep, step_atoms)
        key = input_key(instance.automaton.input_aps, step_input)
        if key not in instance.automaton.transitions[state]:
            raise ValueError(f"missing transition for state={state}, key={key}")
        next_state, out = instance.automaton.transitions[state][key]
        out_step = {k: int(out.get(k, 0)) for k in instance.automaton.output_aps}
        outputs.append(out_step)
        state = next_state
        history.extend(step_atoms)

        effect_now = _effect_triggered_prefix(instance, outputs)
        budget_t_remaining, budget_a_remaining = _remaining_budgets(instance, history)
        obs = canonical_observation(
            timestep,
            outputs,
            effect_now,
            budget_t_remaining,
            budget_a_remaining,
            history,
            instance.mode,
            instance.t_star,
        )
        rendered = _render_observation(obs, renderer_track)
        steps.append(
            {
                "t": int(timestep),
                "action_set": {atom.ap: int(atom.value) for atom in step_atoms},
                "observation": obs,
                "observation_rendered": rendered,
            }
        )
        last_observation = obs

    certificate = sorted_certificate(history)
    verification = evaluate_certificate(instance, certificate)
    # This should always hold because the episode transition path is equivalent
    # to applying the same certificate to the full trace and running once.
    replay_trace = apply_certificate(instance.base_trace, certificate)
    _, replay_outputs = run_automaton(instance.automaton, replay_trace)
    replay_goal = effect_satisfied(instance, replay_outputs)

    return {
        "instance_id": instance.instance_id,
        "mode": instance.mode,
        "t_star": int(instance.t_star),
        "effect_ap": instance.effect_ap,
        "budgets": {
            "timestep": int(instance.budget_timestep),
            "atoms": int(instance.budget_atoms),
        },
        "certificate": [atom.to_tuple() for atom in certificate],
        "steps": steps,
        "suff": bool(verification.suff),
        "min1": bool(verification.min1),
        "valid": bool(verification.valid),
        "goal": bool(verification.goal),
        "replay_goal": bool(replay_goal),
    }
