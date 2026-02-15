# GF-01 Benchmark Harness

This repository contains a runnable reference harness for the GF-01 benchmark
family (human-playable and agent-playable temporal-causality tasks).

## What is in this repo

- `Spec.pdf`: current formal specification snapshot.
- `gf01/`: benchmark runtime, generator, verifier, baselines, and CLI.
- `tests/`: regression tests and fixture artifacts.

## Requirements

- Python 3.10+ (no external dependency install is required for basic use).

## Quick start

Run a single demo instance:

```bash
python3 -m gf01 demo --seed 1337
```

Run a playable episode:

```bash
# human-interactive
python3 -m gf01 play --seed 1337 --renderer-track visual

# baseline agent (non-interactive)
python3 -m gf01 play --seed 1337 --agent greedy --renderer-track json
```

Run checks and tests:

```bash
python3 -m gf01 checks --seed 3000
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Standard benchmark workflow

Generate a bundle + manifest:

```bash
python3 -m gf01 generate \
  --seed 2000 \
  --count 4 \
  --split public_dev \
  --out gf01_public_dev.json \
  --manifest-out gf01_public_dev_manifest.json
```

Evaluate an agent:

```bash
python3 -m gf01 evaluate \
  --instances gf01_public_dev.json \
  --agent greedy \
  --eval-track EVAL-CB \
  --renderer-track json \
  --tool-allowlist-id none \
  --out gf01_runs.jsonl
```

Aggregate and validate:

```bash
python3 -m gf01 report --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json
python3 -m gf01 report --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json --strict
python3 -m gf01 validate --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json --official
```

## Pilot pack freeze (provisional/internal)

Create a deterministic frozen pilot pack:

```bash
python3 -m gf01 freeze-pilot \
  --freeze-id gf01-pilot-freeze-v1 \
  --split pilot_internal_v1 \
  --seed-start 7000 \
  --count 24 \
  --out-dir pilot_freeze/gf01_pilot_freeze_v1
```

## Track policy summary

- `EVAL-CB` (closed-book): no external tool metadata allowed.
- `EVAL-TA` (tool-augmented): requires explicit `tool_allowlist_id` and
  `tool_log_hash`.
- `EVAL-OC` (oracle ceiling): strongest tooling/solver condition, reported
  separately.

`python3 -m gf01 play` enforces these constraints at runtime and logs a
machine-checkable `run_contract` block in output.

## Notes

- Seeds are deterministic provenance inputs.
- Exact normative checks (`suff`, `min1`, `valid`) are enforced in verifier
  paths.
- Run artifacts are schema-versioned and include evaluation-track metadata.
