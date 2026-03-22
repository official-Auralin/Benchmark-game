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

Top-level retained public assets:

- `pilot_freeze/gf01_pilot_freeze_v1`
- `pilot_runs/gf01_pilot_campaign_v1`

Private companion retained artifacts now live outside the primary repo:

- `../gf01_private_companion/source/dev_artifacts/pilot_freeze/hyp018_matched_mode_v4_n240`
- `../gf01_private_companion/source/dev_artifacts/pilot_runs/hyp018_matched_mode_v4_n240`
- `../gf01_private_companion/source/dev_artifacts/pilot_runs/hyp018_trend_summary`
- `../gf01_private_companion/source/dev_artifacts/research_pack_pilot_runs/q033_protocol_v2_mw2`
- `../gf01_private_companion/source/dev_artifacts/research_pack_pilot_runs/q033_calibration_probe_v2`
- `../gf01_private_companion/source/dev_artifacts/research_pack_pilot_runs/release_candidate_smoke_v1`

Older campaign slices and exploratory summaries belong in the external archive.

## Reproducibility Workflow

- Core CLI profile:
  `python3 -m pip install -e .`
- Human-ui profile:
  `python3 -m pip install -e .[human-ui]`
  and `python3 -m pip install -r requirements-human-ui.txt`
- Paper-artifact profile:
  `python3 -m pip install -e .[paper-artifact]`
  and `python3 -m pip install -r requirements-paper-artifact.txt`
- Run `python3 -m gf01 checks --seed 3000` before publication-facing changes.
- Run `python3 -m unittest discover -s tests -p 'test_*.py' -v` before merging.
- Use `python3 -m gf01 gate --fixture-root tests/fixtures/official_example --seed-profile 4200 --unittest-shards 2`
  for the fast gate-equivalent local check.
- Validate the public docs/spec surface with
  `python3 -m unittest tests.test_docs_spec_sync tests.test_repo_layout_policy tests.test_gf01_ci_policy -v`.
- In `../spec_source/`, maintainers rebuild or validate the private TeX source
  with `latexmk -pdf -interaction=nonstopmode -halt-on-error Spec.tex`.
- In `../gf01_private_companion/`, run the private validation suite with
  `python3 -m unittest discover -s tests -p 'test_*.py' -v`.
- For reviewer reproduction, prefer the paper-artifact profile and keep the
  `generate`, `freeze-pilot`, `pilot-campaign`, `release-package`, and
  `q033-build-manifests` outputs deterministic across reruns with identical
  inputs and repo state.

## Q-033 Performance Closure

Use the executable Q-033 command chain, not ad hoc scripts:

- `python3 -m gf01 q033-build-manifests`
- `python3 -m gf01 q033-sweep`
- `python3 -m gf01 q033-closure-check`

Retained committed artifacts should show the manifest, sweep outputs, closure
decision, and hardware declaration needed to defend the result.

## Publication Expectations

- Treat `spec/Spec.pdf` as the normative public benchmark artifact.
- Treat `../spec_source/Spec.tex` as the private TeX authoring source.
- Treat `../gf01_private_companion/skill/gf01-private-companion/references/spec/spec.md`
  as the agent-ingestible companion copy.
- Keep `spec/` and `docs/` in sync with any behavior or policy change.
- Treat `docs/HUMAN_DATA_GOVERNANCE.md` as the current statement of the repo's
  local-only human-study boundary.
- Use `../gf01_private_companion/source/research_pack/26_phase_h2_threats_validity.md`,
  `../gf01_private_companion/source/research_pack/27_phase_h3_ablations_adversarial_checks.md`,
  and `docs/research-notes.md` when preparing paper claims or reviewer
  responses.
