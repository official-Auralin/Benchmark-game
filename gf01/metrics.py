"""
Scoring helpers for comparing agent certificates to benchmark ground truth.

This module computes precision/recall/F1 at two levels: exact intervention
atoms and intervention timesteps. It also resolves ties deterministically when
multiple valid minimal certificates exist for the same instance.
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
from statistics import median

from .models import InterventionAtom, RunRecord
from .semantics import sorted_certificate


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def precision_recall_f1(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def atom_set(certificate: list[InterventionAtom]) -> set[tuple[int, str, int]]:
    return {atom.to_tuple() for atom in certificate}


def timestep_set(certificate: list[InterventionAtom]) -> set[int]:
    return {atom.timestep for atom in certificate}


def _canonical_hash(certificate: list[InterventionAtom]) -> str:
    payload = repr([a.to_tuple() for a in sorted_certificate(certificate)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def best_match_certificate(
    candidate: list[InterventionAtom], minimal_set: list[list[InterventionAtom]]
) -> list[InterventionAtom]:
    if not minimal_set:
        return []
    cand_atoms = atom_set(candidate)
    best = None
    best_key = None
    for gold in minimal_set:
        ap_p, ap_r, ap_f1 = precision_recall_f1(cand_atoms, atom_set(gold))
        _, _, ts_f1 = precision_recall_f1(timestep_set(candidate), timestep_set(gold))
        key = (
            ap_f1,           # maximize
            ts_f1,           # maximize
            -len(timestep_set(gold)),  # prefer lower timestep cost
            _canonical_hash(gold),     # deterministic tie-break
        )
        if best_key is None or key > best_key:
            best_key = key
            best = gold
    return best if best is not None else []


def compute_ap_ts_metrics(
    candidate: list[InterventionAtom], minimal_set: list[list[InterventionAtom]]
) -> tuple[float, float, float, float, float, float]:
    best = best_match_certificate(candidate, minimal_set)
    ap_p, ap_r, ap_f1 = precision_recall_f1(atom_set(candidate), atom_set(best))
    ts_p, ts_r, ts_f1 = precision_recall_f1(timestep_set(candidate), timestep_set(best))
    return ap_p, ap_r, ap_f1, ts_p, ts_r, ts_f1


def aggregate_suite(records: list[RunRecord]) -> dict[str, float]:
    if not records:
        return {}
    certified_rate = sum(1 for r in records if r.valid) / len(records)
    goal_rate = sum(1 for r in records if r.goal) / len(records)
    eff_t_goal = [r.eff_t for r in records if r.goal]
    eff_t_cert = [r.eff_t for r in records if r.valid]
    return {
        "count": float(len(records)),
        "certified_rate": certified_rate,
        "goal_rate": goal_rate,
        "median_eff_t_given_goal": float(median(eff_t_goal)) if eff_t_goal else 0.0,
        "median_eff_t_given_certified": float(median(eff_t_cert)) if eff_t_cert else 0.0,
        "ap_f1_mean": sum(r.ap_f1 for r in records) / len(records),
        "ts_f1_mean": sum(r.ts_f1 for r in records) / len(records),
    }
