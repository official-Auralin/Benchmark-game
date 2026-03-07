# GF-01 Architecture

GF-01 is organized around a small semantic core, thin command wiring, and a
separate documentation backbone that future game families can build on.

## Semantic Core

- `gf01.models`: typed benchmark models and artifact records.
- `gf01.semantics`: canonical trace, observation, and intervention mechanics.
- `gf01.verifier`: exact sufficiency and minimality checks.
- `gf01.generator`: deterministic instance generation.
- `gf01.io`: artifact loading, migration, validation, and writing.
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

- `gf01.renderers.r1_pygame_helpers`: deterministic helper logic that is safe
  to unit test directly.
- `gf01.renderers.r1_pygame`: session flow, event loop, and UI composition.

Renderer work must preserve the observation contract described in
`spec/parity.md`.

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

- Formal normative spec: `spec/tex_files/Spec.tex`
- Operational spec: `spec/`
- Benchmarking and publication context: `docs/benchmarking.md`
- Contributor workflow: `docs/CONTRIBUTING.md`
