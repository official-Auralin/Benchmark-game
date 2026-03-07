# GF-01 Benchmarking

This page is the canonical operator guide for evaluation protocol,
reproducibility, and retained benchmark artifacts.

## Evaluation Protocol

- Keep evaluation tracks separate: `EVAL-CB`, `EVAL-TA`, and `EVAL-OC`.
- Keep renderer tracks explicit. Human-facing runs use `renderer_track=visual`;
  machine-facing runs may use `json` or visual renderers, but scoring semantics
  must remain the same.
- Official validation and reporting must be driven by versioned bundle, run,
  split-manifest, and pilot-freeze artifacts.
- Public and private results must be reported separately; do not pool them.

## Metrics And Reporting

- Primary benchmark success is certified minimal causal validity, not raw goal
  achievement alone.
- Report certified rate, goal rate, AP and timestep precision/recall/F1,
  efficiency summaries, and public/private gaps.
- Keep baseline panels explicit and versioned.
- For publication-facing claims, cite the exact benchmark, generator, checker,
  and policy versions used.

## Retained In-Repo Artifacts

Top-level retained pilot assets:

- `pilot_freeze/gf01_pilot_freeze_v1`
- `pilot_freeze/hyp018_matched_mode_v4_n240`
- `pilot_runs/gf01_pilot_campaign_v1`
- `pilot_runs/hyp018_matched_mode_v4_n240`
- `pilot_runs/hyp018_trend_summary`

Retained research-pack run artifacts:

- `research_pack/pilot_runs/q033_protocol_v2_mw2`
- `research_pack/pilot_runs/q033_calibration_probe_v2`
- `research_pack/pilot_runs/release_candidate_smoke_v1`

Older campaign slices and exploratory summaries belong in the external archive.

## Reproducibility Workflow

- Install contributor dependencies with `python3 -m pip install -r requirements.txt`.
- Run `python3 -m gf01 checks --seed 3000` before publication-facing changes.
- Run `python3 -m unittest discover -s tests -p 'test_*.py' -v` before merging.
- Use `python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2`
  for the fast gate-equivalent local check.
- Rebuild the formal PDF with `python3 scripts/build_spec.py` after TeX or
  formal-spec edits.

## Q-033 Performance Closure

Use the executable Q-033 command chain, not ad hoc scripts:

- `python3 -m gf01 q033-build-manifests`
- `python3 -m gf01 q033-sweep`
- `python3 -m gf01 q033-closure-check`

Retained committed artifacts should show the manifest, sweep outputs, closure
decision, and hardware declaration needed to defend the result.

## Publication Expectations

- Treat `spec/tex_files/Spec.tex` as the formal benchmark authority.
- Keep `spec/` and `docs/` in sync with any behavior or policy change.
- Use `research_pack/26_phase_h2_threats_validity.md`,
  `research_pack/27_phase_h3_ablations_adversarial_checks.md`, and
  `docs/research-notes.md` when preparing paper claims or reviewer responses.
