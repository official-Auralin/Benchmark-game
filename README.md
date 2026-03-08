# GF-01 Benchmark Harness

GF-01 is a benchmark harness for human-playable and agent-playable temporal
causal reasoning tasks. The repository is organized around a formal spec, a
stable reference implementation, retained evaluation artifacts, and a compact
set of canonical docs that future contributors can use without loading the
entire project history.

## Start Here

- `docs/INDEX.md`: primary steering context.
- `spec/overview.md`: purpose, users, scope, and non-goals.
- `spec/contracts.md`: CLI, schema, and policy invariants.
- `spec/environment.md`: task model and observation/action semantics.
- `spec/parity.md`: human-versus-agent information parity rules.
- `docs/benchmarking.md`: evaluation protocol, retained artifact paths, and reproducibility.
- `docs/ARCHITECTURE.md`: module boundaries and extension points.
- `docs/CONTRIBUTING.md`: setup, workflow rules, docs policy, and archive procedure.

## Quick Setup

- Python `3.10+`
- Install mirrored/public-scope dependencies:

```bash
python3 -m pip install -r requirements.txt
```

- Rebuild the formal spec PDF when TeX changes:

```bash
python3 scripts/build_spec.py
```

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

The `pygame` path is a map-first visual layer. It may add readability cues such
as `LIVE`, `TARGET`, and `PIN` sector tags or staging-status badges, but it
must not change the canonical observation contract or scoring behavior. Queued
loadout chips are also interactive removal controls in that visual layer.

Run the standard validation baseline:

```bash
python3 -m gf01 checks --seed 3000
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2
```

## Repo Layout

- `spec/`: operational spec plus the formal PDF and TeX source.
- `docs/`: contributor workflow, architecture, benchmarking, and archive policy.
- `gf01/`: runtime, generator, verifier, baselines, CLI, and renderer code.
- `tests/`: regression suite and fixtures.
- `research_pack/`: retained evidence library only.
- `pilot_freeze/`, `pilot_runs/`: latest retained experiment artifacts.

Historical plans, noisy execution logs, notebooks, and older experiment slices
are archived outside the repo. See `docs/ARCHIVE_LOG.md`.
