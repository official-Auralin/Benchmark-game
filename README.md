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

### Interactive input format (`gf01 play --renderer-track visual`)

At each timestep, the CLI prompt expects an action for the **current timestep
only**.

Accepted forms:

- `skip` (or empty input): make no intervention this step
- `in0=1`: set one input proposition
- `in0=1,in2=0`: set multiple input propositions in the same timestep

Rules:

- Use only AP names shown in `Valid APs` at the prompt.
- Values must be `0` or `1`.
- Do not include timestep labels in the input (no `t=...` prefix).
- Do not use JSON at the interactive prompt.

If you want non-interactive execution (no manual input), use `--agent`:

```bash
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

For matched-seed `HYP-018` comparisons (windowed `normal` vs exact-time `hard`),
freeze two packs with the same seed set and explicit mode overrides:

```bash
SEEDS="7000,7001,7002,7003,7004,7005"

python3 -m gf01 freeze-pilot \
  --freeze-id gf01-hyp018-normal-v1 \
  --split pilot_internal_hyp018 \
  --seeds "$SEEDS" \
  --mode normal \
  --out-dir pilot_freeze/gf01_hyp018_normal_v1

python3 -m gf01 freeze-pilot \
  --freeze-id gf01-hyp018-hard-v1 \
  --split pilot_internal_hyp018 \
  --seeds "$SEEDS" \
  --mode hard \
  --out-dir pilot_freeze/gf01_hyp018_hard_v1
```

Run a pilot campaign on that frozen pack with official validation/report
artifacts:

```bash
python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_pilot_freeze_v1 \
  --out-dir pilot_runs/gf01_pilot_campaign_v1 \
  --baseline-panel random,greedy,search,tool,oracle \
  --renderer-track json \
  --seed 1100
```

Run campaigns for each matched-mode freeze:

```bash
python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_hyp018_normal_v1 \
  --out-dir pilot_runs/gf01_hyp018_normal_campaign_v1 \
  --baseline-panel random,greedy,search,tool,oracle \
  --renderer-track json \
  --seed 1100

python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_hyp018_hard_v1 \
  --out-dir pilot_runs/gf01_hyp018_hard_campaign_v1 \
  --baseline-panel random,greedy,search,tool,oracle \
  --renderer-track json \
  --seed 1100
```

Analyze campaign artifacts against the pre-registered `DEC-014d` trigger checks
(quartile discrimination + shortcut anomaly):

```bash
python3 -m gf01 pilot-analyze \
  --campaign-dir pilot_runs/gf01_pilot_campaign_v1 \
  --eval-track EVAL-CB \
  --mode normal
```

For matched-mode comparisons, analyze each campaign separately:

```bash
python3 -m gf01 pilot-analyze \
  --campaign-dir pilot_runs/gf01_hyp018_normal_campaign_v1 \
  --eval-track EVAL-CB \
  --mode normal

python3 -m gf01 pilot-analyze \
  --campaign-dir pilot_runs/gf01_hyp018_hard_campaign_v1 \
  --eval-track EVAL-CB \
  --mode hard
```

This writes `pilot_analysis.json` inside the campaign directory and prints the
same payload to stdout.

## Deterministic Notebook Campaign (HYP-018)

A reproducible notebook is provided at:

- `research_pack/49_phase_g14_4_hyp018_matched_mode_campaign.ipynb`

The notebook:

- freezes matched-seed `normal` and `hard` packs,
- runs paired campaigns with the same baseline panel,
- computes a deterministic comparison table (`CSV` + `Markdown`),
- and exports SVG charts + HTML report artifacts.

Default artifact output location:

- `pilot_runs/hyp018_matched_mode_v1/artifacts`

## Track policy summary

- `EVAL-CB` (closed-book): no external tool metadata allowed.
- `EVAL-TA` (tool-augmented): requires explicit `tool_allowlist_id` and
  `tool_log_hash`.
- `EVAL-OC` (oracle ceiling): strongest tooling/solver condition, reported
  separately.

`python3 -m gf01 play` enforces these constraints at runtime and logs a
machine-checkable `run_contract` block in output.
The `run_contract` also includes:

- `play_protocol` (`commit_only`)
- `scored_commit_episode` (`true`)

Defaults:

- `play_protocol=commit_only`
- `scored_commit_episode=true`

## Notes

- Seeds are deterministic provenance inputs.
- Exact normative checks (`suff`, `min1`, `valid`) are enforced in verifier
  paths.
- Run artifacts are schema-versioned and include evaluation-track metadata.
