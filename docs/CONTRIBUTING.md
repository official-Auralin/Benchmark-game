# Contributing

## Setup

- Python 3.10+ is required.
- Install public-scope dependencies with `python3 -m pip install -r requirements.txt`.
- `requirements.txt` is intentionally limited to dependencies needed by files
  that are mirrored to the public repo. Do not add local-only, private-only, or
  non-mirrored tooling there.

## Common Commands

- Full tests: `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Baseline checks: `python3 -m gf01 checks --seed 3000`
- Faster gate run: `python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2`
- Build the formal spec PDF: `python3 scripts/build_spec.py`

## Workflow Rules

- Treat `spec/tex_files/Spec.tex` as the formal spec authority.
- Treat `spec/` markdown pages as the compact implementation steering context.
- Prefer small, local refactors that preserve existing public behavior.
- Add or update characterization tests before large structural changes.

## Refactor Checklist

- Confirm the public contract in `spec/contracts.md` before editing code.
- Prefer changes that stay within one command module and one core module.
- If a small feature would touch more than two code modules, treat that as a
  design smell and extract a boundary first.
- Keep parser wiring in `gf01.cli_registry`, command entrypoints in
  `gf01.commands.*`, and benchmark logic in core modules such as `gf01.io`,
  `gf01.generator`, `gf01.play`, and `gf01.q033`.
- Keep deterministic renderer helpers in `gf01.renderers.r1_pygame_helpers`
  and session/UI flow in `gf01.renderers.r1_pygame`.
- Run targeted tests for the affected command family before the full suite.
- Re-run `python3 -m gf01 checks --seed 3000` after command-wiring or policy
  changes.

## Change Locality Target

- Baseline before this refactor: small CLI or policy changes commonly touched
  `3-4` code modules.
- Target after this refactor: most small command or policy changes should touch
  no more than `2` code modules, plus the matching spec/test updates.

## Docs Freshness Checklist

- If you changed CLI flags or behavior, update `README.md` and `spec/contracts.md`.
- If you changed schema or policy identifiers, update `spec/contracts.md` and
  relevant tests.
- If you changed architecture or module boundaries, update
  `docs/ARCHITECTURE.md`.
- If you changed benchmark behavior, update `spec/acceptance-tests.md` and the
  formal spec if needed.
- If you changed mirrored dependency needs, update `requirements.txt`, but only
  for dependencies required by mirrored/public files.
