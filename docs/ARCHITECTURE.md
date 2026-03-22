# GF-01 Architecture

GF-01 is organized around a small semantic core, thin command wiring, and a
separate documentation backbone that future game families can build on.

## Semantic Core

- `gf01.models`: typed benchmark models and artifact records.
- `gf01.semantics`: canonical trace, observation, and intervention mechanics.
- `gf01.verifier`: exact sufficiency and minimality checks.
- `gf01.generator`: deterministic instance generation.
- `gf01.formal_loader`: raw formal-artifact ingestion and normalization.
- `gf01.io`: canonical bundle loading, migration, validation, and artifact writing.
- `gf01.meta`: schema versions, policy versions, and reproducibility metadata.

These modules define the stable implementation layer that later game families
should reuse whenever possible.

## Workflow Layer

- `gf01.play`: play loop and policy adapters.
- `gf01.checks`, `gf01.profiling`, `gf01.q033`, `gf01.gate`: benchmark-quality
  and profiling workflows.
- `gf01.commands.workflows`: generation, evaluation, reporting, validation, and migration commands.
- `gf01.commands.quality`: checks, profiling, governance, release, and identifiability commands.
- `gf01.commands.pilot`, `gf01.commands.p0`, `gf01.commands.q033`, `gf01.commands.playback`:
  pilot-study, P0, q033, and interactive-play command groups.
- `gf01.cli_registry`: grouped parser registration.
- `gf01.cli`: public CLI entrypoint and compatibility layer.

## Renderer Boundary

- `gf01.renderers.r1_theme`: deterministic mapping from formal benchmark
  elements to AP-grounded causal-board labels, icons, and colors.
- `gf01.renderers.r1_grid`: deterministic spatial layout derived from control/
  signal roles and coarse relation structure, plus timeline helpers.
- `gf01.renderers.r1_pygame`: session flow, event loop, parity-safe inspector,
  and UI composition for the canonical `GF-01-R1` causal-board window.

Renderer work must preserve the observation contract described in
`spec/parity.md`.

Official semantic identity is normalized formal content, not `seed`. Legacy
seed-backed generation remains available as development tooling, but the active
runtime now also supports direct raw formal ingestion via `--formal`.

## Dependency Rules

- Parser construction belongs in `gf01.cli_registry`.
- Command entrypoints belong in `gf01.commands.*`.
- Benchmark behavior belongs in the semantic core and workflow modules, not in
  parser wiring.
- Dependencies should point inward toward stable semantics, not outward toward
  UI-specific code.

## Extension Direction

Future game families should aim to reuse:

- the schema and reporting layer,
- the validation and governance machinery,
- the benchmarking protocol,
- and as much of the semantic core as the family contract allows.

When a new family requires new behavior, prefer adding a local module boundary
over spreading edits through multiple command groups.

## Steering Documents

- Normative public spec: `spec/Spec.pdf`
- Private spec authoring source: `../spec_source/Spec.tex`
- Private agent skill: `../gf01_private_companion/skill/gf01-private-companion/`
- Operational spec: `spec/`
- Benchmarking and publication context: `docs/benchmarking.md`
- Contributor workflow: `docs/CONTRIBUTING.md`
