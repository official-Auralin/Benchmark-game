# GF-01 Architecture

GF-01 is organized around a small semantic core plus command-oriented workflow
layers.

## Core Modules

- `gf01.models`: typed benchmark data models and run-record types.
- `gf01.semantics`: canonical trace, observation, and rendering mechanics.
- `gf01.verifier`: certificate checking and exact validation logic.
- `gf01.generator`: deterministic instance generation.
- `gf01.io`: external artifact loading, migration, validation, and writing.
- `gf01.meta`: schema versions, policy versions, and reproducibility metadata.

## Workflow Modules

- `gf01.play`: playable episode loop and policy adapters.
- `gf01.checks`, `gf01.profiling`, `gf01.q033`, `gf01.gate`: benchmark-quality
  and profiling workflows.
- `gf01.commands.workflows`: generation, evaluation, report, validation, and migration commands.
- `gf01.commands.quality`: checks, profiling, governance, release, and identifiability commands.
- `gf01.commands.pilot`, `gf01.commands.p0`, `gf01.commands.q033`, `gf01.commands.playback`:
  pilot-study, P0, profiling, and interactive-play command groups.
- `gf01.cli_registry`: parser registration grouped by command family.
- `gf01.cli`: public CLI entrypoint and compatibility layer.
- `gf01.renderers.r1_pygame_helpers`: deterministic visual-backend helpers kept separate from session flow.

## Architectural Direction

- Keep public parser construction in one place.
- Keep command implementations grouped by workflow domain.
- Keep business logic in the core modules above, not in parser wiring.
- Keep renderer helper logic pure and separately testable from UI/session flow.

## Steering Documents

- Formal normative spec: `spec/tex_files/Spec.tex`
- Operational steering spec: `spec/`
- Contributor workflow and freshness rules: `docs/CONTRIBUTING.md`
