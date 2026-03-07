# GF-01 Benchmark Harness

This repository contains a runnable reference harness for the GF-01 benchmark
family (human-playable and agent-playable temporal-causality tasks).

## What is in this repo

- `spec/Spec.pdf`: current formal specification snapshot.
- `spec/tex_files/Spec.tex`: formal normative source for the benchmark spec.
- `spec/`: compact operational spec for contracts and acceptance scenarios.
- `docs/ARCHITECTURE.md`: system map and module-boundary guide.
- `docs/CONTRIBUTING.md`: setup, test, spec-build, and docs-freshness workflow.
- `requirements.txt`: contributor dependencies.
- `gf01/`: benchmark runtime, generator, verifier, baselines, and CLI.
- `tests/`: regression tests and fixture artifacts.

## Requirements

- Python 3.10+.
- Install mirrored/public-scope dependencies with `python3 -m pip install -r requirements.txt`.
- Optional for graphical human play: `pygame-ce` is the only current external dependency.

## Contributor entrypoints

- Use `docs/CONTRIBUTING.md` for setup, test commands, and documentation freshness rules.
- Use `docs/ARCHITECTURE.md` for system boundaries before changing command wiring.
- Use `spec/contracts.md` and `spec/acceptance-tests.md` as the compact behavior contract.
- Rebuild the formal spec PDF with `python3 scripts/build_spec.py`.

## Quick start

Run a single demo instance:

```bash
python3 -m gf01 demo --seed 1337
```

Run a playable episode:

```bash
# human-interactive (terminal visual snapshot)
python3 -m gf01 play --seed 1337 --renderer-track visual --visual-backend text

# human-interactive (graphical map-first window)
python3 -m gf01 play --seed 1337 --renderer-track visual --visual-backend pygame

# baseline agent (non-interactive)
python3 -m gf01 play --seed 1337 --agent greedy --renderer-track json
```

### Interactive play modes (`gf01 play --renderer-track visual`)

`--visual-backend text` (default):
- Uses terminal snapshots and typed action input.

`--visual-backend pygame`:
- Opens a graphical map-first window.
- Click per-AP buttons (`0` or `1`) to set interventions for the current step.
- Click `clear` on an AP to unset it for this step.
- Press `Enter` to commit the step.
- Press `Esc` to skip the step.
- Press `Backspace` to clear all current-step selections.
- Press `Left`/`Right` (or `PageUp`/`PageDown`) to page through AP controls
  on high-AP instances.

### Terminal input format (`--visual-backend text`)

At each timestep, the CLI prompt expects an action for the **current timestep
only**.

Accepted forms:

- `skip` (or empty input): make no intervention this step
- `in0=1`: set one input proposition
- `in0=1,in2=0`: set multiple input propositions in the same timestep

Visual renderer notes (both backends):

- `--renderer-track visual` now shows a structured snapshot (time/mode/effect,
  timeline rail, budgets, current outputs, intervention history) to make human
  play easier to read in terminal sessions.
- `--visual-backend pygame` renders the map-first interactive view with
  intervention controls and an `Output delta` summary so players can see what
  changed since the prior observed state.
- In `pygame` mode, the observation panel shows `Previous command` alongside
  `Output delta` so players can map the last committed input changes to the
  newly observed response.
- In `pygame` mode, objective status is shown as a color-coded badge
  (`Objective active` / `Objective not active`) for faster at-a-glance reading.
- In `pygame` mode, a short onboarding strip appears for timesteps `t=0..2`
  to guide first-time players through action, feedback reading, and adjustment.
- In `pygame` mode, a `Sector Wave Strip (observed)` panel shows output-activity
  pressure and short trend text (`baseline`/`rising`/`steady`/`falling`) using
  only observed outputs.
- The same wave panel includes a short recent-trend trail (`t=k:trend`) to
  make temporal pattern shifts readable across the last few timesteps.
- The wave panel also shows a `Hot sectors` summary (top recent sectors by
  observed pressure tokens) to support faster map-first triage on long traces.
- In `pygame` mode, a `Command -> Sector response (observed)` trail shows
  recent per-step command summaries paired with observed output deltas.
- The pygame observation panel now includes plain summaries for
  currently-active observed outputs and your pending interventions to clarify
  action->response interpretation without formal notation.
- In `pygame` mode, timeline sectors now include explicit `N`/`T`/`B` marks
  (`now`, `target`, `both`) in addition to color coding.
- In `pygame` mode, the objective window is explicitly labeled and highlighted
  on the timeline (`hard`: exact `t*`; `normal`: window `t* - w .. t*`).
- In `pygame` mode, timeline text now includes an objective-window pressure
  coverage summary (`observed/total`, peak sector token, and average pressure).
- In `pygame` mode, `[`/`]` zoom the visible timeline window (narrow/wide) so
  long traces remain readable without shrinking each sector cell.
- In `pygame` mode, a compact timeline minimap strip shows the full horizon and
  current viewport window at a glance (`[` and `]` mark the visible range).
- In `pygame` mode, a sampled rectangular `Sector board` panel shows the full
  horizon as a grid (viewport-highlighted cells, objective-window tint, and
  `N/T/B` markers for now/target/both).
- In `pygame` mode, hover a sector-board cell to inspect the represented
  timestep bucket (`t` range, pressure token, edit token, and marker).
- In `pygame` mode, the `Mission window` card now explains whether the hovered
  sector contains the live turn, the target timestep, overlaps the scoring
  window, or sits outside it.
- In `pygame` mode, sector-board hover also links to the timeline: the
  corresponding timestep range gets a mint border in the timeline row.
- In `pygame` mode, the sector-board border shows a short command-focus trail:
  green = most recent command bucket, cyan = previous, blue = third-most-recent.
  This links command and observed-response views without exposing hidden state.
- In `pygame` mode, the sector board includes grid coordinates (`A..H`, `1..6`)
  and compact glyphs (`p`, `e`, `*`) so non-experts can parse state changes
  quickly while keeping the same canonical information.
- In `pygame` mode, closing the window aborts the interactive run (it is not
  treated as a `skip`/no-op action).
- In `pygame` mode, keys `1..9` and `0` cycle the corresponding visible AP
  control on the current page through `unset -> 1 -> 0 -> unset`.
- In `pygame` mode, `+`/`-` adjusts AP page density (`page_size`) to trade off
  more controls per page vs larger per-control readability.
- In `pygame` mode, `G` cycles AP-group focus (`ALL -> group1 -> group2 ...`)
  so high-AP levels can be explored one group at a time.
- In `pygame` mode, `C` toggles collapsible map rows; when collapse is on and
  group=`ALL`, AP rows are collapsed until a specific group is selected.
- In `pygame` mode, `H` toggles an in-session quick-help overlay with control
  reminders and action->effect reading tips.
- In `pygame` mode, `I` toggles a canonical observation inspector that shows
  mission metadata plus the full canonical observation payload (`O(s)`) for
  parity with agent-visible information.
- The pygame control header also shows a compact AP-group summary for the
  current page (e.g., `groups: in(6), sensor(4)`).
- The timeline rail includes:
  - `mark` row (`N` = now, `T` = target, `B` = now and target are same step)
  - `pressure` band inside each sector cell (dim = lower observed output
    activity, bright = higher observed output activity)
  - `P` token row (`P0..P10`) for numeric pressure readout per visible sector
  - `edits` row (count of interventions applied at each timestep so far)
- Canonical scoring semantics are unchanged.

Rules:

- Use only AP names shown in `Valid APs` at the prompt.
- Values must be `0` or `1`.
- Do not include timestep labels in the input (no `t=...` prefix).
- Do not use JSON at the interactive prompt.

If you want non-interactive execution (no manual input), use `--agent`:

```bash
python3 -m gf01 play --seed 1337 --agent greedy --renderer-track json
```

### P0 internal-alpha feedback gate

Use `p0-feedback-check` to convert internal playthrough feedback CSV files into
a machine-checkable pass/fail decision.

One-shot setup (recommended) to create both the feedback template and the
deterministic P0 seed pack:

```bash
python3 -m gf01 p0-init \
  --template-out p0_feedback.csv \
  --seeds 7000,7001,7002,7003,7004,7005,7006,7007 \
  --out-dir research_pack/pilot_freeze/gf01_p0_alpha_v1 \
  --force
```

If you prefer explicit step-by-step setup, use the two commands below.

Create a deterministic starter template:

```bash
python3 -m gf01 p0-feedback-template --out p0_feedback.csv
```

Create a deterministic P0 seed pack (default split/mode/profile):

```bash
python3 -m gf01 p0-seed-pack \
  --seeds 7000,7001,7002,7003,7004,7005,7006,7007 \
  --out-dir research_pack/pilot_freeze/gf01_p0_alpha_v1 \
  --force
```

Required CSV columns:
- `tester_id`
- `objective_clarity`
- `control_clarity`
- `action_effect_clarity`
- `must_fix_blockers`

Example:

```bash
python3 -m gf01 p0-session-check \
  --feedback p0_feedback.csv \
  --runs-dir p0_runs \
  --required-renderer-track visual \
  --out p0_session_summary.json

python3 -m gf01 p0-feedback-check \
  --feedback p0_feedback.csv \
  --min-score 3 \
  --min-ratio 0.80 \
  --out p0_feedback_summary.json
```

`p0-session-check` expects play artifacts named
`p0_runs/<tester_id>_<seed>.json` (matching `seed_list_run` from feedback CSV).

Combined gate (recommended after sessions complete):

```bash
python3 -m gf01 p0-gate \
  --feedback p0_feedback.csv \
  --runs-dir p0_runs \
  --required-renderer-track visual \
  --min-score 3 \
  --min-ratio 0.80 \
  --out p0_gate_summary.json
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
- `--visual-backend pygame` is valid only with `--renderer-track visual`.

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
- `visual_backend` (`text` or `pygame`)

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
