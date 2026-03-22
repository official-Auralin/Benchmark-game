# GF-01 Benchmark Harness

GF-01 is a benchmark harness for human-playable and agent-playable temporal
causal reasoning tasks. This repository is now the primary public contract
surface: it carries the stable implementation, the normative published spec
PDF, the retained public benchmark artifacts, and the compact contributor docs.

This repository is released under the Apache-2.0 license.

## Start Here

- `docs/INDEX.md`: primary steering context.
- `spec/overview.md`: purpose, users, scope, and non-goals.
- `spec/contracts.md`: CLI, schema, and policy invariants.
- `spec/environment.md`: task model and observation/action semantics.
- `spec/parity.md`: human-versus-agent information parity rules.
- `docs/benchmarking.md`: evaluation protocol, retained artifact paths, and reproducibility.
- `docs/HUMAN_DATA_GOVERNANCE.md`: current local-only human-data boundary and future deployment checklist.
- `docs/ARCHITECTURE.md`: module boundaries and extension points.
- `docs/CONTRIBUTING.md`: setup, workflow rules, docs policy, and archive procedure.
- `docs/LOCAL_COMPANION.md`: maintainer workflow for `../spec_source/` and `../gf01_private_companion/`.

## Quick Setup

- Python `3.10+`
- Core profile:

```bash
python3 -m pip install -e .
```

- Human-ui profile:

```bash
python3 -m pip install -e .[human-ui]
python3 -m pip install -r requirements-human-ui.txt
```

- Paper-artifact profile:

```bash
python3 -m pip install -e .[paper-artifact]
python3 -m pip install -r requirements-paper-artifact.txt
```

- Maintainers who update the formal spec or private evidence chain also work in:
  - `../spec_source/` for editable `Spec.tex`
  - `../gf01_private_companion/` for private research/evidence and the agent skill
- The primary repo publishes and validates `spec/Spec.pdf`. Editable TeX
  authoring lives outside this repo at `../spec_source/Spec.tex`.

## Quick Commands

Run a single demo instance:

```bash
python3 -m gf01 demo --seed 1337
```

Run a playable episode:

```bash
python3 -m gf01 play --seed 1337 --renderer-track visual --visual-backend text
python3 -m gf01 play --seed 1337 --renderer-track visual --visual-backend pygame
python3 -m gf01 play --seed 1337 --agent greedy --renderer-track json
```

The `pygame` path is the canonical `GF-01-R1` tower-defense graphical window.
It may add readability cues such as defense cards, wave timelines, staged
deployment highlights, and causal feedback overlays, but it must not change the
canonical observation contract or scoring behavior.

Run the standard validation baseline:

```bash
python3 -m gf01 checks --seed 3000
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2
```

## Repo Layout

- `spec/`: operational spec plus the normative public `spec/Spec.pdf`.
- `docs/`: contributor workflow, architecture, benchmarking, and archive policy.
- `gf01/`: runtime, generator, verifier, baselines, CLI, and renderer code.
- `tests/`: regression suite and fixtures.
- `pilot_freeze/gf01_pilot_freeze_v1`: retained public freeze artifact.
- `pilot_runs/gf01_pilot_campaign_v1`: retained public campaign artifact.
- `../spec_source/`: private TeX authoring repo for maintainers.
- `../gf01_private_companion/`: private evidence repo and structured agent skill.

Historical plans, mirror-era tooling, noisy execution logs, and superseded
experiment slices are archived outside the repo. See `docs/ARCHIVE_LOG.md`.
