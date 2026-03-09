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
  the mirrored/public files. Do not add local-only, private-only, or
  non-mirrored tooling there.
- `latexmk` is required for the private source repo paper-artifact path and for
  `scripts/build_spec.py`.
- The public GitHub mirror intentionally omits `spec/tex_files/Spec.tex`; the
  TeX source is locked and not contributor-editable there.

## Canonical Doc Map

- `docs/INDEX.md`: first stop for contributors and agents.
- `spec/tex_files/Spec.tex`: formal normative spec authority.
- `spec/contracts.md`: public contract and version-policy page.
- `spec/environment.md`: environment model and semantics.
- `spec/parity.md`: human-versus-agent information parity.
- `spec/plan.md`: only active long-lived plan.
- `docs/benchmarking.md`: reproducibility and evaluation protocol.
- `docs/HUMAN_DATA_GOVERNANCE.md`: current human-data boundary and future deployment checklist.
- `docs/ARCHITECTURE.md`: code boundaries and extension points.
- `docs/STYLE.md`: documentation hygiene rules.

## Common Commands

- Full tests:
  `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Baseline checks:
  `python3 -m gf01 checks --seed 3000`
- Faster gate-equivalent run:
  `python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2`
- Build the formal spec PDF in the private source repo:
  `python3 scripts/build_spec.py`
- Check that the committed formal spec PDF is fresh in the private source repo:
  `python3 scripts/build_spec.py --check`
- Public-mirror spec validation:
  `python3 -m unittest tests.test_docs_spec_sync tests.test_spec_tex_integrity -v`
- Human-ui smoke path:
  `SDL_VIDEODRIVER=dummy python3 -m unittest tests.test_gf01_play_loop tests.test_r1_renderer_modules -v`

## Workflow Rules

- Prefer small, local refactors that preserve existing public behavior.
- Update the spec before or alongside any behavior change.
- Add or update characterization tests before large structural changes.
- Do not create new active plan documents when `spec/plan.md` can be updated.
- Treat `research_pack/` as an evidence library, not the default contributor
  navigation surface.

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
  formal spec if needed.
- If you changed parity or renderer behavior, update `spec/parity.md`.
- If you changed architecture or module boundaries, update
  `docs/ARCHITECTURE.md`.
- If you changed benchmarking protocol or retained artifact policy, update
  `docs/benchmarking.md`.
- If you changed mirrored dependency needs, update `requirements.txt`, but only
  for dependencies required by mirrored/public files. Update the profile-specific
  requirements files and `pyproject.toml` at the same time when applicable.
