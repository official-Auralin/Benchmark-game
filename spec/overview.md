# GF-01 Overview

GF-01 is a benchmark harness for human-playable and agent-playable
temporal-causality tasks. It provides deterministic instance generation,
play/evaluation workflows, machine-checkable scoring artifacts, and release
governance checks so benchmark results can be reproduced and audited.

The operational spec is split across:

- `spec/contracts.md`
- `spec/environment.md`
- `spec/parity.md`
- `spec/acceptance-tests.md`
- `spec/plan.md`

## Users

- Benchmark maintainers publishing validated GF-01 releases.
- Researchers evaluating baseline or external agents on frozen instances.
- Human-study operators running internal pilot and P0 play sessions.
- Contributors extending the reference harness without changing the public
  contract.

## Scope

- Generate deterministic GF-01 instance bundles and manifests.
- Run benchmark checks, profiling, evaluation, reporting, and validation.
- Support human and agent play through the `gf01 play` contract.
- Produce machine-checkable pilot, release, and governance artifacts.

## Non-goals

- Local/private research notes and unpublished workflows are not part of the
  mirrored public dependency surface.
- The markdown spec in `spec/` does not replace the formal TeX specification in
  `spec/tex_files/Spec.tex`; it is the compact operational steering layer.
