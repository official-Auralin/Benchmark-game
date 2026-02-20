"""
Identifiability diagnostics for partial-observability generation policies.

This module provides a deterministic metric that estimates whether passive
observations plus intervention consequences are sufficiently informative for a
generated instance. It does not score agents directly; it is used to accept or
reject low-signal instances during generation and to report policy metadata.
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

from typing import Any

from .meta import (
    IDENTIFIABILITY_METRIC_ID,
    IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
    IDENTIFIABILITY_POLICY_VERSION,
)
from .models import GF01Instance
from .semantics import apply_certificate, run_automaton
from .verifier import candidate_atoms


def _observable_signature(instance: GF01Instance, outputs: list[dict[str, int]]) -> tuple[tuple[int, ...], ...]:
    output_aps = sorted(instance.automaton.output_aps)
    return tuple(
        tuple(int(step.get(ap, 0)) for ap in output_aps)
        for step in outputs
    )


def instance_identifiability_metrics(instance: GF01Instance) -> dict[str, Any]:
    """
    Compute deterministic identifiability diagnostics for one instance.

    Metric logic:
    - enumerate all single-atom intervention candidates (one edit at one
      timestep),
    - compare each resulting observable output trace against baseline,
    - summarize response and diversity in observable signatures.
    """
    _, base_outputs = run_automaton(instance.automaton, instance.base_trace)
    base_sig = _observable_signature(instance, base_outputs)
    atoms = candidate_atoms(instance)
    if not atoms:
        return {
            "policy_version": IDENTIFIABILITY_POLICY_VERSION,
            "metric_id": IDENTIFIABILITY_METRIC_ID,
            "candidate_atom_count": 0,
            "response_atom_count": 0,
            "response_ratio": 0.0,
            "unique_signature_count": 1,
            "signature_diversity_ratio": 0.0,
            "passes_threshold": False,
        }

    signatures: list[tuple[tuple[int, ...], ...]] = []
    response_count = 0
    for atom in atoms:
        trace = apply_certificate(instance.base_trace, [atom])
        _, outputs = run_automaton(instance.automaton, trace)
        sig = _observable_signature(instance, outputs)
        signatures.append(sig)
        if sig != base_sig:
            response_count += 1

    candidate_count = len(atoms)
    unique_count = len(set(signatures))
    response_ratio = response_count / float(candidate_count)
    diversity_ratio = unique_count / float(candidate_count)
    passes = (
        response_ratio >= IDENTIFIABILITY_MIN_RESPONSE_RATIO
        and unique_count >= IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES
    )
    return {
        "policy_version": IDENTIFIABILITY_POLICY_VERSION,
        "metric_id": IDENTIFIABILITY_METRIC_ID,
        "candidate_atom_count": candidate_count,
        "response_atom_count": response_count,
        "response_ratio": response_ratio,
        "unique_signature_count": unique_count,
        "signature_diversity_ratio": diversity_ratio,
        "passes_threshold": bool(passes),
    }


def identifiability_policy_error(
    metrics: dict[str, Any],
    *,
    min_response_ratio: float = IDENTIFIABILITY_MIN_RESPONSE_RATIO,
    min_unique_signatures: int = IDENTIFIABILITY_MIN_UNIQUE_SIGNATURES,
) -> str | None:
    response_ratio = float(metrics.get("response_ratio", 0.0))
    unique_count = int(metrics.get("unique_signature_count", 0))
    if response_ratio < float(min_response_ratio):
        return (
            f"response_ratio={response_ratio:.6f} below threshold "
            f"{float(min_response_ratio):.6f}"
        )
    if unique_count < int(min_unique_signatures):
        return (
            f"unique_signature_count={unique_count} below threshold "
            f"{int(min_unique_signatures)}"
        )
    return None
