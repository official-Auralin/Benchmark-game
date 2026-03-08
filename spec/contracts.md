# GF-01 Public Contracts

`gf01` exposes its primary public contract through the `python -m gf01` CLI,
versioned JSON/JSONL artifacts, and the formal benchmark semantics in
`spec/tex_files/Spec.tex`.

## Command Contract

- Public entrypoint: `python -m gf01`.
- Stable command families: generation/evaluation/reporting, validation and
  migration, release/governance checks, q033 workflows, pilot workflows,
  playable episodes, and P0 gates.
- This curation pass does not change command names, flags, exit codes, or
  machine output shapes.

## Artifact Schemas

- `gf01.instance_bundle.v1`
- `gf01.run_record.v1`
- `gf01.split_manifest.v1`
- `gf01.pilot_freeze.v1`

These schema identifiers are defined in `gf01.meta` and must remain aligned
with code, tests, and documentation.

## Policy Versions

- `gf01.adaptation_policy.v1`
- `gf01.renderer_policy.v1`
- `gf01.identifiability_policy.v1`
- `gf01.complexity_policy.v1`
- `gf01.baseline_panel_policy.v1`
- `gf01.split_policy.v1`
- `gf01.rotation_policy.v1`
- `gf01.tool_policy.v1`

## Invariants

- Generation is deterministic for fixed inputs.
- Track separation remains strict across `EVAL-CB`, `EVAL-TA`, and `EVAL-OC`.
- Strict validation remains the guard for official or release-grade artifacts.
- Renderer, adaptation, tool-allowlist, split, rotation, and baseline-panel
  policies remain machine-checkable and versioned.
- Public mirror contents remain intentionally minimal.
- `requirements.txt` is public-scope-only; local/private dependencies must not
  be added there.

## Public Mirror Contract

The public mirror is synced from an explicit allowlist and currently includes:

- `README.md`
- `requirements.txt`
- `docs/`
- `spec/Spec.pdf`
- `spec/overview.md`
- `spec/contracts.md`
- `spec/environment.md`
- `spec/parity.md`
- `spec/acceptance-tests.md`
- `spec/plan.md`
- `.github/workflows/gf01-gate.yml`
- `gf01/`
- `tests/`

Private research materials, local sync tooling, and local-only dependencies are
out of scope for the public mirror.
