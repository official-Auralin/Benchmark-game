"""Q-033 command parser registration."""

from __future__ import annotations

import argparse

from .commands.q033 import (
    cmd_q033_build_manifests,
    cmd_q033_closure_check,
    cmd_q033_sweep,
)


def register_q033_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_q033_manifest = sub.add_parser(
        "q033-build-manifests",
        help="Build deterministic balanced quartile seed manifests for Q-033 sweeps",
    )
    p_q033_manifest.add_argument("--seed-start", type=int, default=8000)
    p_q033_manifest.add_argument("--candidate-count", type=int, default=4000)
    p_q033_manifest.add_argument("--replicates", type=int, default=2)
    p_q033_manifest.add_argument("--per-quartile", type=int, default=120)
    p_q033_manifest.add_argument("--split", type=str, default="q033_internal")
    p_q033_manifest.add_argument("--out-dir", type=str, default="q033_manifests/q033_protocol_v1")
    p_q033_manifest.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty output directory",
    )
    p_q033_manifest.set_defaults(func=cmd_q033_build_manifests)

    p_q033_sweep = sub.add_parser(
        "q033-sweep",
        help="Run one Q-033 profiling sweep replicate from a seed manifest",
    )
    p_q033_sweep.add_argument("--manifest", type=str, required=True)
    p_q033_sweep.add_argument("--baseline-panel", type=str, default="random,greedy,search,tool,oracle")
    p_q033_sweep.add_argument("--seed", type=int, default=1300)
    p_q033_sweep.add_argument("--max-generate-ms-mean", type=float, default=1200.0)
    p_q033_sweep.add_argument("--max-minset-ms-mean", type=float, default=2500.0)
    p_q033_sweep.add_argument("--max-eval-ms-mean", type=float, default=1500.0)
    p_q033_sweep.add_argument("--max-checks-total-ms", type=float, default=30000.0)
    p_q033_sweep.add_argument("--max-truncation-rate", type=float, default=0.25)
    p_q033_sweep.add_argument("--min-oracle-minus-greedy-gap", type=float, default=0.10)
    p_q033_sweep.add_argument("--max-quartile-truncation-rate", type=float, default=0.30)
    p_q033_sweep.add_argument("--min-quartile-gap", type=float, default=0.05)
    p_q033_sweep.add_argument("--max-quartile-runtime-gate-failures", type=int, default=1)
    p_q033_sweep.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_q033_sweep.set_defaults(func=cmd_q033_sweep)

    p_q033_close = sub.add_parser(
        "q033-closure-check",
        help="Check Q-033 closure rule from two or more sweep outputs",
    )
    p_q033_close.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Path to one q033-sweep output JSON (repeat flag for multiple replicates)",
    )
    p_q033_close.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help="Disable disjoint-seed requirement across replicates",
    )
    p_q033_close.add_argument("--out", type=str, default="", help="Optional output JSON path")
    p_q033_close.set_defaults(func=cmd_q033_closure_check)
