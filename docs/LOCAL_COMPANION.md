# Local Companion Workflow

This repo is the primary public contract surface. Maintainers who work on the
private spec authoring source or the private evidence chain also keep two
sibling git repos:

- `../spec_source/`
- `../gf01_private_companion/`

## Roles

- `../spec_source/Spec.tex`: authoritative private TeX authoring source
- `../spec_source/sources.bib`: synchronized build copy for `Spec.tex`
- `../gf01_private_companion/source/research_pack/01_sources/sources.bib`:
  authoritative bibliography inventory
- `../gf01_private_companion/source/research_pack/`: private evidence library
- `../gf01_private_companion/skill/gf01-private-companion/`: structured agent
  skill for maintainers and AI agents

## Audit Trail

- `../spec_source/migration_ledger.json` records the source repo SHA and
  subtree split used to create the spec-authoring repo.
- `../gf01_private_companion/migration_ledger.json` records the source repo SHA
  and subtree split used to create the private companion repo.
- `../archive/Research/2026-03-22_repo-simplification/02_public-repo-history/public_repo.bundle`
  preserves the retired nested `public_repo/` git history.

## Required Linked Changes

Every formal spec update must keep these three artifacts synchronized:

- `../spec_source/Spec.tex`
- `spec/Spec.pdf`
- `../gf01_private_companion/skill/gf01-private-companion/references/spec/spec.md`

## Validation

- Primary repo public surface:
  `python3 -m unittest tests.test_docs_spec_sync tests.test_repo_layout_policy tests.test_gf01_ci_policy -v`
- `../spec_source/` authoring check:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error Spec.tex`
- `../gf01_private_companion/` validation suite:
  `python3 -m unittest discover -s tests -p 'test_*.py' -v`

## CI Status Contexts

- `GF01 Gate / gate`
- `GF01 Gate / release-candidate`
