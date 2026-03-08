# GF-01 Active Plan

GF-01 is moving toward a stable benchmark backbone for temporal causal
reasoning: a fixed formal specification, a reusable implementation layer for
future game families, a human-playable interface with agent parity, and a
publication-ready evidence package for benchmark results.

## Active Priorities

1. Keep the formal spec, operational spec, and runtime behavior aligned.
2. Preserve a stable implementation layer that future families can extend
   without changing GF-01's public contract.
3. Improve human-playable clarity while preserving canonical information parity.
4. Run reproducible pilot and benchmark workflows with retained artifact trails.
5. Maintain paper-quality rationale, threats, ablations, and governance.

## Execution Rules

- `spec/plan.md` is the only active planning authority.
- Do not create new long-lived phase-plan files when this page can be updated.
- Keep active guidance in `spec/` and `docs/`; move superseded operational
  narratives to the external archive.
- Every meaningful behavior, policy, or interface change must update tests and
  the relevant canonical docs in the same change.

## Immediate Workstreams

- Spec backbone: keep `spec/tex_files/Spec.tex`, `spec/contracts.md`,
  `spec/environment.md`, and `spec/parity.md` synchronized.
- Stable layer: keep `gf01` modular enough that future family work stays local.
- Human parity: keep the `GF-01-R1` visual path readable without hidden-state
  leakage.
- Benchmark ops: keep gate, profiling, q033, pilot, and release workflows
  reproducible.
- Publication: keep the retained evidence chain defensible and the archive
  organized.
