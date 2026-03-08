# Archive Log

Historical material is archived outside the repo so the active working set
stays small.

## 2026-03-07 Repo Curation

- Archive root:
  `../archive/Research/2026-03-07_repo-curation/`
- Purpose:
  move superseded plans, operational logs, notebooks, older experiment slices,
  and presentation outputs out of the active repo without deleting history.
- Additional archived design aids:
  the unused visual mockup gallery and its generator script now live under
  `01_superseded-canonical-docs/visual-parity/`.

Archive categories:

- `01_superseded-canonical-docs/`
- `02_operational-history/`
- `03_analysis-and-presentations/`
- `04_experiment-artifacts/`

Within those categories, the current batch is further split into:

- `benchmark-core/`, `master-plans/`, `phase-history/`, `visual-parity/`
- `public-mirror-pushes/`, `post-merge-and-rechecks/`, `regression-guards/`,
  `review-resolutions/`
- `analysis-exports/`, `notebooks/`, `summaries/`,
  `legacy-spec-snapshots/`
- `pilot_freeze/hyp018/`, `pilot_runs/hyp018/`, and
  `research_pack_pilot_runs/`

Each top-level archive category must contain a short `README.md` describing
what moved there and why.

## Active In-Repo Set After Curation

- `README.md`
- `docs/`
- `spec/`
- `gf01/`
- `tests/`
- retained `research_pack/` evidence backbone
- retained latest pilot and run artifacts
