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

Visual renderer notes:

- `--renderer-track visual` now shows a structured snapshot (time/mode/effect,
  budgets, current outputs, intervention history) to make human play easier to
  read in terminal sessions.
- This is still a CLI renderer (not a graphical game window); canonical scoring
  semantics are unchanged.

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
  --baseline-policy-level full \
  --renderer-track json \
  --seed 1100
```

To merge human/external play sessions produced by `play --out`, pass
`--external-episodes` (repeatable). Accepted formats are JSON (single payload
or list of payloads) and JSONL (one payload per line):

```bash
python3 -m gf01 play \
  --instances pilot_freeze/gf01_pilot_freeze_v1/instance_bundle_v1.json \
  --instance-index 0 \
  --script demo_actions.json \
  --renderer-track json \
  --out pilot_runs/external_episode_000.json

python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_pilot_freeze_v1 \
  --out-dir pilot_runs/gf01_pilot_campaign_v1_plus_external \
  --baseline-panel random,greedy,oracle \
  --baseline-policy-level core \
  --renderer-track json \
  --seed 1100 \
  --external-episodes pilot_runs/external_episode_000.json
```

Run campaigns for each matched-mode freeze:

```bash
python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_hyp018_normal_v1 \
  --out-dir pilot_runs/gf01_hyp018_normal_campaign_v1 \
  --baseline-panel random,greedy,search,tool,oracle \
  --baseline-policy-level full \
  --renderer-track json \
  --seed 1100

python3 -m gf01 pilot-campaign \
  --freeze-dir pilot_freeze/gf01_hyp018_hard_v1 \
  --out-dir pilot_runs/gf01_hyp018_hard_campaign_v1 \
  --baseline-panel random,greedy,search,tool,oracle \
  --baseline-policy-level full \
  --renderer-track json \
  --seed 1100
```

Baseline-panel policy (`gf01.baseline_panel_policy.v1`):

- `--baseline-policy-level full` (default): requires
  `random,greedy,search,tool,oracle`.
- `--baseline-policy-level core`: requires `random,greedy,oracle` and is
  intended for faster internal smoke runs.

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

Build a reproducibility package from frozen+campaign artifacts (strict
validation is enforced before packaging):

```bash
python3 -m gf01 release-package \
  --freeze-dir pilot_freeze/gf01_pilot_freeze_v1 \
  --campaign-dir pilot_runs/gf01_pilot_campaign_v1 \
  --out-dir release_packages/gf01_release_package_v1
```

This emits `release_package_manifest.json`, `RERUN_INSTRUCTIONS.md`, and an
`artifacts/` directory with pinned bundle/manifest/runs/report files.

`pilot-analyze` now emits `complexity_policy_version=gf01.complexity_policy.v1`
with machine-checkable complexity diagnostics:

- composite quartile summaries (held-out greedy and pooled evaluation scope),
- per-knob diagnostics (`*_knob_stats`) with `is_constant` flags,
- per-agent knob slices (`per_agent_knob_stats`) for auditability.

If historical `runs_combined.jsonl` rows predate current required metadata
fields, `pilot-analyze` applies deterministic in-memory migration before strict
validation and records the operation under `legacy_migration` in output.

## Deterministic HYP-018 artifacts (CLI-first)

Use freeze + pilot campaign + pilot analyze commands to generate deterministic
paired-mode artifacts (windowed `normal` vs exact-time `hard`) from explicit
seed lists.

Typical output locations:

- paired run artifacts: `pilot_runs/hyp018_matched_mode_*/artifacts`
- trend summaries (if you aggregate slices): `pilot_runs/hyp018_trend_summary/`

If you maintain a private research notebook workflow, keep it as an optional
presentation layer over these same command outputs.

## Q-033 high-performance closure workflow

The pre-registered `Q-033` protocol is executable via three commands.

1) Build deterministic balanced quartile seed manifests:

```bash
python3 -m gf01 q033-build-manifests \
  --seed-start 8000 \
  --candidate-count 4000 \
  --replicates 2 \
  --per-quartile 120 \
  --split q033_internal \
  --out-dir q033_manifests/q033_protocol_v1
```

2) Run one sweep per replicate manifest (default thresholds are protocol values
baked into the CLI for `gf01.q033_protocol.v1`):

```bash
python3 -m gf01 q033-sweep \
  --manifest q033_manifests/q033_protocol_v1/q033-rep-01.json \
  --out q033_runs/q033_rep01_sweep.json

python3 -m gf01 q033-sweep \
  --manifest q033_manifests/q033_protocol_v1/q033-rep-02.json \
  --out q033_runs/q033_rep02_sweep.json
```

3) Apply closure rule across replicates:

```bash
python3 -m gf01 q033-closure-check \
  --sweep q033_runs/q033_rep01_sweep.json \
  --sweep q033_runs/q033_rep02_sweep.json \
  --out q033_runs/q033_closure_report.json
```

`q033-closure-check` exits nonzero unless all replicate gates pass and seed sets
are disjoint (unless `--allow-seed-overlap` is explicitly supplied).

## Track policy summary

- `EVAL-CB` (closed-book): no external tool metadata allowed.
- `EVAL-TA` (tool-augmented): requires explicit `tool_allowlist_id` and
  `tool_log_hash`.
- `EVAL-OC` (oracle ceiling): strongest tooling/solver condition, reported
  separately.

Allowed `tool_allowlist_id` values:

- `EVAL-CB`: `none`
- `EVAL-TA`: `local-planner-v1`
- `EVAL-OC`: `oracle-exact-search-v1`

`python3 -m gf01 play` enforces these constraints at runtime and logs a
machine-checkable `run_contract` block in output.

Renderer policy (`gf01.renderer_policy.v1`) is also enforced:

- `renderer_track=json` requires `renderer_profile_id=canonical-json-v1`.
- `renderer_track=visual` requires `renderer_profile_id=GF-01-R1`.
- No other renderer profiles are accepted in current official runtime paths.

Adaptation/fine-tuning metadata policy (`gf01.adaptation_policy.v1`) is also
enforced for `play`, `evaluate`, `pilot-campaign`, and `migrate-runs`:

- `no_adaptation` requires:
  - `adaptation_budget_tokens=0`
  - `adaptation_data_scope=none`
  - `adaptation_protocol_id=none`
- `prompt_adaptation` or `weight_finetune` requires:
  - `adaptation_budget_tokens>0`
  - `adaptation_data_scope` in `public_only` or `public_plus_external`
  - non-empty `adaptation_protocol_id`

Relevant CLI flags:

- `--adaptation-condition {no_adaptation,prompt_adaptation,weight_finetune}`
- `--adaptation-budget-tokens <int>`
- `--adaptation-data-scope {none,public_only,public_plus_external}`
- `--adaptation-protocol-id <string>`

Example adapted condition:

```bash
python3 -m gf01 evaluate \
  --seed 1337 \
  --eval-track EVAL-TA \
  --tool-allowlist-id local-planner-v1 \
  --tool-log-hash demo_tool_log_hash \
  --adaptation-condition weight_finetune \
  --adaptation-budget-tokens 5000 \
  --adaptation-data-scope public_only \
  --adaptation-protocol-id ft-public-v1
```

The `run_contract` also includes:

- `play_protocol` (`commit_only`)
- `scored_commit_episode` (`true`)
- `renderer_policy_version` (`gf01.renderer_policy.v1`)
- `renderer_profile_id` (`canonical-json-v1` for JSON track, `GF-01-R1` for visual track)

Defaults:

- `play_protocol=commit_only`
- `scored_commit_episode=true`

## Split governance policy check

For publication governance, run split-ratio policy checks on a split manifest:

```bash
python3 -m gf01 split-policy-check \
  --manifest split_manifest_v1.json \
  --target-ratios public_dev=0.2,public_val=0.2,private_eval=0.6 \
  --tolerance 0.05 \
  --private-split private_eval \
  --min-private-eval-count 1 \
  --require-official-split-names \
  --strict-manifest
```

This emits a machine-checkable JSON report and exits nonzero on policy
violations.

To enforce release-cycle rotation and contamination safeguards against a prior
manifest, run:

```bash
python3 -m gf01 release-governance-check \
  --manifest split_manifest_current_v1.json \
  --previous-manifest split_manifest_previous_v1.json \
  --require-previous-manifest \
  --target-ratios public_dev=0.2,public_val=0.2,private_eval=0.6 \
  --tolerance 0.05 \
  --private-split private_eval \
  --min-private-eval-count 1 \
  --min-public-novelty-ratio 0.10
```

This checks both:
- split policy (`gf01.split_policy.v1`), and
- release rotation policy (`gf01.rotation_policy.v1`):
  - no previous-private instances are exposed in current public splits,
  - current public split has enough novel instances versus previous public.

For release reporting policy (required baseline panel + per-track/per-slice
coverage), run:

```bash
python3 -m gf01 release-report-check \
  --runs runs_combined.jsonl \
  --manifest split_manifest_v1.json \
  --baseline-policy-level full
```

This enforces `gf01.baseline_panel_policy.v1` coverage by manifest slice and
fails if required baseline agents/tracks are missing from release artifacts.

To execute governance + report + package checks in one deterministic command,
run:

```bash
python3 -m gf01 release-candidate-check \
  --freeze-dir pilot_freeze/gf01_pilot_freeze_v1 \
  --campaign-dir pilot_runs/gf01_pilot_campaign_v1 \
  --baseline-policy-level full \
  --package-out-dir release_packages/gf01_release_candidate_v1 \
  --out release_candidate_check.json
```

Optional rotation hardening flags for the integrated command:
- `--previous-manifest <path>`
- `--require-previous-manifest`
- `--min-public-novelty-ratio <float>`

Use `--skip-package` if you want to validate governance/report stages without
building a release package in the same invocation.

## Identifiability policy check (partial observability)

To validate that an instance bundle satisfies the locked identifiability policy
(`gf01.identifiability_policy.v1`), run:

```bash
python3 -m gf01 identifiability-check \
  --instances tests/fixtures/official_example/instance_bundle_v1.json \
  --min-response-ratio 0.60 \
  --min-unique-signatures 8
```

This emits a machine-checkable JSON report and exits nonzero if any instance
falls below threshold.

## Notes

- Seeds are deterministic provenance inputs.
- Exact normative checks (`suff`, `min1`, `valid`) are enforced in verifier
  paths.
- Run artifacts are schema-versioned and include evaluation-track metadata.
