# GF-01 Public Contracts

`gf01` exposes its primary public contract through the `python -m gf01` CLI,
versioned JSON/JSONL artifacts, and the formal benchmark semantics published in
`spec/Spec.pdf`.

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
- The primary repo remains the public contract surface.
- `requirements.txt` is primary-repo-only; local/private dependencies must not
  be added there.
- Runtime defaults must not depend on `research_pack/`, `public_repo/`, or
  sibling private paths.

## Repo Contract

The canonical machine-readable inventory of the primary repo surface is
`spec/contract_inventory.json`. It defines the exact public docs, workflow,
runtime code, tests, and retained public artifacts that belong in this repo.

`docs/LOCAL_COMPANION.md` is the authoritative maintainer reference for the
expected sibling-repo topology and linked-change workflow.

Private authoring and evidence surfaces live outside this repo:

- `../spec_source/Spec.tex`: private TeX authoring source
- `../gf01_private_companion/skill/gf01-private-companion/references/spec/spec.md`:
  agent-ingestible companion copy
- `../gf01_private_companion/source/research_pack/`: private evidence library

`research_pack/`, `spec/tex_files/`, `public_repo/`, and local sync/build
tooling are out of scope for the primary repo.
