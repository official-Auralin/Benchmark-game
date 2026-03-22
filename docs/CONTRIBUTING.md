# Contributing

## Setup

- Python `3.10+` is required.
- Install the core editable package with:
  `python3 -m pip install -e .`
- Install the public human-ui profile with:
  `python3 -m pip install -e .[human-ui]`
  and `python3 -m pip install -r requirements-human-ui.txt`
- Install the paper-artifact profile with:
  `python3 -m pip install -e .[paper-artifact]`
  and `python3 -m pip install -r requirements-paper-artifact.txt`
- `requirements.txt` remains intentionally limited to dependencies needed by
  the primary repo runtime surface. Do not add local-only, private-only, or
  sibling-companion tooling there.
- `latexmk` is only required in `../spec_source/` when maintainers rebuild the
  private TeX authoring source.

## Canonical Doc Map

- `docs/INDEX.md`: first stop for contributors and agents.
- `spec/Spec.pdf`: normative public spec surface.
- `spec/contracts.md`: public contract and version-policy page.
- `spec/environment.md`: environment model and semantics.
- `spec/parity.md`: human-versus-agent information parity.
- `spec/plan.md`: only active long-lived plan.
- `docs/benchmarking.md`: reproducibility and evaluation protocol.
- `docs/HUMAN_DATA_GOVERNANCE.md`: current human-data boundary and future deployment checklist.
- `docs/ARCHITECTURE.md`: code boundaries and extension points.
- `docs/STYLE.md`: documentation hygiene rules.
- `docs/LOCAL_COMPANION.md`: sibling-repo maintainer workflow.

## Common Commands

- Full tests:
  `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Baseline checks:
  `python3 -m gf01 checks --seed 3000`
- Faster gate-equivalent run:
  `python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2`
- Public docs/spec surface validation:
  `python3 -m unittest tests.test_docs_spec_sync tests.test_repo_layout_policy tests.test_gf01_ci_policy -v`
- Human-ui smoke path:
  `SDL_VIDEODRIVER=dummy python3 -m unittest tests.test_gf01_play_loop tests.test_r1_renderer_modules -v`
- Private spec authoring check in `../spec_source/`:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error Spec.tex`
- Private companion validation in `../gf01_private_companion/`:
  `python3 -m unittest discover -s tests -p 'test_*.py' -v`

## Workflow Rules

- Prefer small, local refactors that preserve existing public behavior.
- Update the spec before or alongside any behavior change.
- Add or update characterization tests before large structural changes.
- Do not create new active plan documents when `spec/plan.md` can be updated.
- Treat the private companion as the evidence library, not the default
  contributor navigation surface for this repo.

## Change Locality Target

- Baseline before the CLI refactor: small CLI or policy changes commonly
  touched `3-4` code modules.
- Current target: most small command or policy changes should touch no more
  than `2` code modules, plus the matching spec/test updates.

## Documentation Policy

- Extend a canonical doc before adding a new top-level doc.
- Keep one source of truth per topic. Use `docs/STYLE.md` if unsure where a
  change belongs.
- If a document is no longer needed for active development or paper
  defensibility, archive it instead of keeping it in-tree.

## Archive Procedure

- Move non-critical historical material to a dated batch under
  `../archive/Research/`.
- Organize the archive by purpose, not by one flat list of files.
- Add a short entry to `docs/ARCHIVE_LOG.md` whenever a new archive batch is
  created.

## Docs Freshness Checklist

- If you changed CLI flags or behavior, update `README.md` and `spec/contracts.md`.
- If you changed schema or policy identifiers, update `spec/contracts.md` and
  relevant tests.
- If you changed environment semantics, update `spec/environment.md` and the
  normative spec artifacts when needed.
- If you changed parity or renderer behavior, update `spec/parity.md`.
- If you changed architecture or module boundaries, update
  `docs/ARCHITECTURE.md`.
- If you changed benchmarking protocol or retained artifact policy, update
  `docs/benchmarking.md`.
- If you changed the formal spec, update these three artifacts together:
  `../spec_source/Spec.tex`, `spec/Spec.pdf`, and
  `../gf01_private_companion/skill/gf01-private-companion/references/spec/spec.md`.
- If you changed primary-repo dependency needs, update `requirements.txt`, the
  profile-specific requirements files, and `pyproject.toml` together when
  applicable.
