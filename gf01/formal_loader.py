"""
Raw formal-artifact ingestion and normalization for GF-01.

This module turns raw formal task packages into validated GF-01 instances. The
supported path is intentionally explicit: either provide a direct Mealy-style
automaton object, or provide a deterministic HOA transition graph plus
state-output annotations and a finite base trace. The normalization step
computes canonical content hashes and rejects malformed or non-total task
artifacts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .meta import (
    ALLOWED_MODES,
    FORMAL_BUNDLE_SCHEMA_VERSION,
    FORMAL_TASK_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    stable_hash_json,
)
from .models import GF01Instance, MealyAutomaton
from .semantics import all_input_valuations, input_key


_EDGE_RE = re.compile(r"^\[(?P<label>.+)\]\s+(?P<target>\d+)$")
_STATE_RE = re.compile(r'^State:\s+(?P<idx>\d+)(?:\s+"(?P<name>[^"]+)")?')


class FormalArtifactError(ValueError):
    """Raised when a raw formal artifact cannot be normalized safely."""


def _load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise FormalArtifactError(f"empty formal artifact file: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _load_trace(payload: dict[str, Any], base_dir: Path) -> list[dict[str, int]]:
    if "base_trace" in payload:
        raw = payload["base_trace"]
    elif "trace_path" in payload:
        raw = _load_json_or_jsonl(base_dir / str(payload["trace_path"]))
    else:
        raise FormalArtifactError("formal task is missing base_trace/trace_path")
    if not isinstance(raw, list) or not raw:
        raise FormalArtifactError("base_trace must be a non-empty list")
    trace: list[dict[str, int]] = []
    for step in raw:
        if not isinstance(step, dict):
            raise FormalArtifactError("base_trace steps must be objects")
        trace.append({str(k): int(v) for k, v in step.items()})
    return trace


def _tokenize_formula(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()!&|":
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < len(text) and (text[j].isalnum() or text[j] in {"_", "."}):
            j += 1
        if j == i:
            raise FormalArtifactError(f"unsupported HOA token near: {text[i:]!r}")
        tokens.append(text[i:j])
        i = j
    return tokens


def _eval_formula(formula: str, ap_names: list[str], valuation: dict[str, int]) -> bool:
    tokens = _tokenize_formula(formula)
    pos = 0

    def _parse_or() -> bool:
        nonlocal pos
        value = _parse_and()
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            value = value or _parse_and()
        return value

    def _parse_and() -> bool:
        nonlocal pos
        value = _parse_not()
        while pos < len(tokens) and tokens[pos] == "&":
            pos += 1
            value = value and _parse_not()
        return value

    def _parse_not() -> bool:
        nonlocal pos
        if pos < len(tokens) and tokens[pos] == "!":
            pos += 1
            return not _parse_not()
        return _parse_primary()

    def _parse_primary() -> bool:
        nonlocal pos
        if pos >= len(tokens):
            raise FormalArtifactError(f"incomplete HOA formula: {formula!r}")
        token = tokens[pos]
        if token == "(":
            pos += 1
            value = _parse_or()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise FormalArtifactError(f"unclosed HOA formula: {formula!r}")
            pos += 1
            return value
        pos += 1
        lowered = token.lower()
        if lowered == "t":
            return True
        if lowered == "f":
            return False
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(ap_names):
                raise FormalArtifactError(f"HOA AP index out of range: {idx}")
            return bool(int(valuation.get(ap_names[idx], 0)))
        if token not in valuation:
            raise FormalArtifactError(f"unknown HOA AP token: {token}")
        return bool(int(valuation[token]))

    value = _parse_or()
    if pos != len(tokens):
        raise FormalArtifactError(f"unused HOA tokens in formula: {formula!r}")
    return value


def _parse_hoa_subset(text: str) -> dict[str, Any]:
    ap_names: list[str] = []
    start_state: int | None = None
    state_names: dict[int, str] = {}
    edges: dict[int, list[tuple[str, int]]] = {}
    current_state: int | None = None
    in_body = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "--BODY--":
            in_body = True
            continue
        if line == "--END--":
            break
        if not in_body:
            if line.startswith("AP:"):
                ap_names = re.findall(r'"([^"]+)"', line)
            elif line.startswith("Start:"):
                parts = line.split(":", 1)[1].strip()
                start_state = int(parts)
            continue
        state_match = _STATE_RE.match(line)
        if state_match:
            current_state = int(state_match.group("idx"))
            state_names[current_state] = state_match.group("name") or f"s{current_state}"
            edges.setdefault(current_state, [])
            continue
        edge_match = _EDGE_RE.match(line)
        if edge_match and current_state is not None:
            edges[current_state].append(
                (edge_match.group("label").strip(), int(edge_match.group("target")))
            )

    if start_state is None:
        raise FormalArtifactError("HOA is missing Start")
    if not ap_names:
        raise FormalArtifactError("HOA is missing AP names")
    if not state_names:
        raise FormalArtifactError("HOA is missing states")
    return {
        "ap_names": ap_names,
        "start_state": start_state,
        "state_names": state_names,
        "edges": edges,
    }


def _state_output_map(
    output_by_state_raw: dict[str, Any],
    *,
    state_names: dict[int, str],
    output_aps: list[str],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for state_idx, state_name in state_names.items():
        payload = None
        if str(state_idx) in output_by_state_raw:
            payload = output_by_state_raw[str(state_idx)]
        elif state_name in output_by_state_raw:
            payload = output_by_state_raw[state_name]
        if not isinstance(payload, dict):
            raise FormalArtifactError(
                f"missing output_by_state mapping for state {state_name}"
            )
        valuation = {str(k): int(v) for k, v in payload.items()}
        if sorted(valuation) != sorted(output_aps):
            raise FormalArtifactError(
                f"output_by_state for {state_name} must match output_aps exactly"
            )
        out[state_name] = valuation
    return out


def _automaton_from_hoa(payload: dict[str, Any], base_dir: Path) -> MealyAutomaton:
    hoa_text = str(payload.get("hoa", "")).strip()
    hoa_path = str(payload.get("hoa_path", "")).strip()
    if not hoa_text:
        if not hoa_path:
            raise FormalArtifactError("formal HOA task requires hoa or hoa_path")
        hoa_text = (base_dir / hoa_path).read_text(encoding="utf-8")
    parsed = _parse_hoa_subset(hoa_text)
    input_aps = [str(ap) for ap in payload.get("input_aps", [])]
    output_aps = [str(ap) for ap in payload.get("output_aps", [])]
    if not input_aps or not output_aps:
        raise FormalArtifactError("formal HOA task requires input_aps and output_aps")
    output_by_state_raw = payload.get("output_by_state")
    if not isinstance(output_by_state_raw, dict):
        raise FormalArtifactError("formal HOA task requires output_by_state")
    state_outputs = _state_output_map(
        output_by_state_raw,
        state_names=parsed["state_names"],
        output_aps=output_aps,
    )

    transitions: dict[str, dict[str, tuple[str, dict[str, int]]]] = {}
    valuations = all_input_valuations(input_aps)
    for state_idx, state_name in parsed["state_names"].items():
        state_edges = parsed["edges"].get(state_idx, [])
        transitions[state_name] = {}
        for valuation in valuations:
            matches = [
                target_idx
                for label, target_idx in state_edges
                if _eval_formula(label, parsed["ap_names"], valuation)
            ]
            if len(matches) != 1:
                raise FormalArtifactError(
                    "HOA normalization requires deterministic total input coverage "
                    f"for state {state_name}; valuation {valuation} matched {len(matches)} edges"
                )
            next_state = parsed["state_names"][matches[0]]
            transitions[state_name][input_key(input_aps, valuation)] = (
                next_state,
                dict(state_outputs[next_state]),
            )

    return MealyAutomaton(
        states=[parsed["state_names"][idx] for idx in sorted(parsed["state_names"])],
        initial_state=parsed["state_names"][parsed["start_state"]],
        input_aps=input_aps,
        output_aps=output_aps,
        transitions=transitions,
    )


def _normalize_automaton_payload(payload: dict[str, Any], base_dir: Path) -> MealyAutomaton:
    automaton_raw = payload.get("automaton")
    if isinstance(automaton_raw, dict):
        transitions: dict[str, dict[str, tuple[str, dict[str, int]]]] = {}
        for state, edges in automaton_raw["transitions"].items():
            transitions[str(state)] = {}
            for key, edge_payload in edges.items():
                transitions[str(state)][str(key)] = (
                    str(edge_payload["next_state"]),
                    {str(k): int(v) for k, v in edge_payload["output"].items()},
                )
        return MealyAutomaton(
            states=[str(s) for s in automaton_raw["states"]],
            initial_state=str(automaton_raw["initial_state"]),
            input_aps=[str(a) for a in automaton_raw["input_aps"]],
            output_aps=[str(a) for a in automaton_raw["output_aps"]],
            transitions=transitions,
        )
    return _automaton_from_hoa(payload, base_dir)


def _validate_instance_fields(
    *,
    automaton: MealyAutomaton,
    base_trace: list[dict[str, int]],
    effect_ap: str,
    t_star: int,
    mode: str,
    window_size: int,
    budget_timestep: int,
) -> None:
    input_aps = list(automaton.input_aps)
    output_aps = list(automaton.output_aps)
    if len(set(input_aps)) != len(input_aps):
        raise FormalArtifactError("input_aps must be unique")
    if len(set(output_aps)) != len(output_aps):
        raise FormalArtifactError("output_aps must be unique")
    if set(input_aps) & set(output_aps):
        raise FormalArtifactError("input_aps and output_aps must be disjoint")
    if effect_ap not in output_aps:
        raise FormalArtifactError("effect_ap must belong to output_aps")
    if mode not in ALLOWED_MODES:
        raise FormalArtifactError(f"mode must be one of {list(ALLOWED_MODES)}")
    if t_star < 0 or t_star >= len(base_trace):
        raise FormalArtifactError("t_star out of range")
    if window_size < 1:
        raise FormalArtifactError("window_size must be >= 1")
    if budget_timestep < 1 or budget_timestep > len(base_trace):
        raise FormalArtifactError("budget_timestep must be in 1..len(base_trace)")

    expected_keys = sorted(input_aps)
    valuations = {
        input_key(input_aps, valuation): valuation
        for valuation in all_input_valuations(input_aps)
    }
    for t, step in enumerate(base_trace):
        if sorted(step) != expected_keys:
            raise FormalArtifactError(
                f"base_trace[{t}] must match input_aps exactly"
            )
        for ap, value in step.items():
            if int(value) not in (0, 1):
                raise FormalArtifactError(
                    f"base_trace[{t}][{ap}] must be binary"
                )

    for state in automaton.states:
        if state not in automaton.transitions:
            raise FormalArtifactError(f"missing transition map for state {state}")
        state_edges = automaton.transitions[state]
        if sorted(state_edges) != sorted(valuations):
            raise FormalArtifactError(
                f"state {state} does not cover the input-total legal domain"
            )
        for key, (next_state, out) in state_edges.items():
            if next_state not in automaton.states:
                raise FormalArtifactError(
                    f"transition from {state} targets unknown state {next_state}"
                )
            if sorted(out) != sorted(output_aps):
                raise FormalArtifactError(
                    f"transition {state}/{key} output valuation must match output_aps exactly"
                )
            for out_ap, value in out.items():
                if int(value) not in (0, 1):
                    raise FormalArtifactError(
                        f"transition {state}/{key} output {out_ap} must be binary"
                    )


def _semantic_payload(
    *,
    automaton: MealyAutomaton,
    base_trace: list[dict[str, int]],
    effect_ap: str,
    t_star: int,
    mode: str,
    window_size: int,
    budget_timestep: int,
) -> dict[str, Any]:
    return {
        "automaton": automaton.to_canonical_dict(),
        "base_trace": [
            {k: int(step[k]) for k in sorted(step)} for step in base_trace
        ],
        "effect_ap": str(effect_ap),
        "t_star": int(t_star),
        "mode": str(mode),
        "window_size": int(window_size),
        "budget_timestep": int(budget_timestep),
    }


def instance_from_formal_dict(
    payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> GF01Instance:
    base_dir = base_dir or Path(".")
    automaton = _normalize_automaton_payload(payload, base_dir)
    base_trace = _load_trace(payload, base_dir)
    effect_ap = str(payload["effect_ap"])
    t_star = int(payload["t_star"])
    mode = str(payload["mode"])
    window_size = int(payload.get("window_size", 1))
    budget_timestep = int(payload["budget_timestep"])

    _validate_instance_fields(
        automaton=automaton,
        base_trace=base_trace,
        effect_ap=effect_ap,
        t_star=t_star,
        mode=mode,
        window_size=window_size,
        budget_timestep=budget_timestep,
    )

    semantic_payload = _semantic_payload(
        automaton=automaton,
        base_trace=base_trace,
        effect_ap=effect_ap,
        t_star=t_star,
        mode=mode,
        window_size=window_size,
        budget_timestep=budget_timestep,
    )
    content_hash = stable_hash_json(semantic_payload)
    provenance = {
        "content_hash": content_hash,
        "normalization_version": NORMALIZATION_VERSION,
        "source_type": str(payload.get("source_type", "formal_artifact")),
        "source_id": str(payload.get("source_id", payload.get("artifact_id", ""))),
    }
    if isinstance(payload.get("provenance"), dict):
        provenance.update({str(k): v for k, v in payload["provenance"].items()})
        provenance["content_hash"] = content_hash
        provenance["normalization_version"] = str(
            provenance.get("normalization_version", NORMALIZATION_VERSION)
        )
    if payload.get("hoa_path"):
        provenance["hoa_path"] = str(payload["hoa_path"])
    if payload.get("tlsf_path"):
        provenance["tlsf_path"] = str(payload["tlsf_path"])

    return GF01Instance(
        instance_id=str(payload.get("instance_id") or f"gf01-{content_hash[:12]}"),
        automaton=automaton,
        base_trace=base_trace,
        effect_ap=effect_ap,
        t_star=t_star,
        mode=mode,
        window_size=window_size,
        budget_timestep=budget_timestep,
        budget_atoms=(
            int(payload["budget_atoms"])
            if payload.get("budget_atoms") is not None
            else None
        ),
        seed=(
            int(payload["seed"])
            if payload.get("seed") is not None
            else None
        ),
        complexity={str(k): float(v) for k, v in payload.get("complexity", {}).items()},
        identifiability={
            str(k): v for k, v in payload.get("identifiability", {}).items()
        },
        split_id=str(payload.get("split_id", "public_dev")),
        renderer_track=str(payload.get("renderer_track", "json")),
        provenance=provenance,
    )


def load_formal_bundle(path: str) -> tuple[list[GF01Instance], dict[str, Any]]:
    src = Path(path)
    if src.is_dir():
        task_path = src / "task.json"
        if not task_path.exists():
            raise FormalArtifactError(
                f"formal artifact directory {src} must contain task.json"
            )
        payload = _load_json_or_jsonl(task_path)
        base_dir = src
    elif src.suffix.lower() == ".hoa":
        candidates = [src.with_suffix(".task.json"), src.with_suffix(".json")]
        task_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if task_path is None:
            raise FormalArtifactError(
                f"direct HOA ingestion requires a sidecar task JSON next to {src.name}"
            )
        payload = _load_json_or_jsonl(task_path)
        if isinstance(payload, dict):
            payload = {**payload, "hoa_path": src.name}
        base_dir = src.parent
    else:
        payload = _load_json_or_jsonl(src)
        base_dir = src.parent

    bundle_meta: dict[str, Any] = {
        "schema_version": FORMAL_BUNDLE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "source_path": str(src),
    }
    if isinstance(payload, list):
        raw_instances = payload
    elif isinstance(payload, dict):
        bundle_meta["source_schema_version"] = str(
            payload.get("schema_version", FORMAL_TASK_SCHEMA_VERSION)
        )
        if "instances" in payload:
            raw_instances = payload["instances"]
            bundle_meta.update(
                {
                    str(k): v
                    for k, v in payload.items()
                    if k != "instances"
                }
            )
        else:
            raw_instances = [payload]
    else:
        raise FormalArtifactError("formal artifact payload must be an object or list")
    if not isinstance(raw_instances, list):
        raise FormalArtifactError("formal artifact instances must be a list")

    instances = []
    for item in raw_instances:
        if not isinstance(item, dict):
            raise FormalArtifactError("formal artifact instances must be objects")
        instances.append(instance_from_formal_dict(item, base_dir=base_dir))
    return instances, bundle_meta
