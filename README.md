# GF-01 Harness (Implementation Kickoff)

This repository now includes an executable first-pass harness for the GF-01 benchmark specification:

- formal core: `research_pack/21_phase_g1_gf01_formal_spec.md`
- generator + exact checks: `research_pack/22_phase_g2_generator_ground_truth.md`
- metrics/baselines/repro: `research_pack/23_phase_g3_metrics_baselines_repro.md`
- consolidated full spec: `research_pack/24_phase_g4_full_spec_gf01.md`

## Quick start

Run single-instance demo:

```bash
python3 -m gf01 demo --seed 1337
```

Run one playable episode:

```bash
# human-interactive episode
python3 -m gf01 play --seed 1337 --renderer-track visual

# baseline-agent episode (non-interactive)
python3 -m gf01 play --seed 1337 --agent greedy --renderer-track json

# tool-augmented track episode (requires explicit tool metadata)
python3 -m gf01 play --seed 1337 --agent tool --eval-track EVAL-TA \
  --tool-allowlist-id local-planner-v1 --tool-log-hash demo-log-hash \
  --renderer-track json
```

Run priority H3 checks (including structural-invalid edge cases):

```bash
python3 -m gf01 checks --seed 3000
```

Run Python-first profiling + performance gates:

```bash
python3 -m gf01 profile --seed 4000 --public-count 3 --private-count 3
```

Optional detailed profiler artifact:

```bash
python3 -m gf01 profile --seed 4000 --cprofile-out gf01_profile.stats
```

Generate a small suite as a versioned instance bundle and write a split
manifest:

```bash
python3 -m gf01 generate --seed 2000 --count 4 --split public_dev --out gf01_public_dev.json --manifest-out gf01_public_dev_manifest.json
```

Evaluate one baseline on an instance file and write per-run JSONL artifacts:

```bash
python3 -m gf01 evaluate --instances gf01_public_dev.json --agent greedy --eval-track EVAL-CB --renderer-track json --tool-allowlist-id none --out gf01_runs.jsonl
```

Aggregate run artifacts by `(eval_track, renderer_track, split_id, mode)` and
optionally compare coverage against a split manifest:

```bash
python3 -m gf01 report --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json
```

Run strict schema validation during reporting (non-zero exit on violations):

```bash
python3 -m gf01 report --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json --strict
```

Run official reporting mode (strict + manifest required):

```bash
python3 -m gf01 report --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json --official
```

Run standalone validation without aggregation output:

```bash
python3 -m gf01 validate --runs gf01_runs.jsonl --manifest gf01_public_dev_manifest.json --official
```

Backfill legacy/pre-v1 run JSONL into v1 schema, then re-validate:

```bash
python3 -m gf01 migrate-runs --runs legacy_runs.jsonl --out migrated_runs.jsonl --manifest gf01_public_dev_manifest.json
python3 -m gf01 validate --runs migrated_runs.jsonl --manifest gf01_public_dev_manifest.json --strict
```

Build a manifest from any existing instance file:

```bash
python3 -m gf01 manifest --instances gf01_public_dev.json --out gf01_manifest.json
```

Run fixture-based regression tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Run one-shot CI-style regression gate:

```bash
python3 -m gf01 gate --fixture-root tests/fixtures/official_example
```

GitHub Actions CI is defined in:

```text
.github/workflows/gf01-gate.yml
```

It runs the same gate on push/PR and uploads `gf01_gate_summary.json` as an
artifact for inspection.

## Notes

- `seed` is deterministic provenance input (no runtime entropy in generation).
- exact normative checks (sufficiency + singleton-removal minimality) are enforced in verifier paths.
- optional stronger diagnostics are modeled as policy fields; full cap-governance integration is staged for pilot calibration.
- Instance outputs support both legacy list format and versioned bundle format (`--legacy-list` keeps list-only output).
- JSONL run artifacts include schema/version metadata, track metadata, certificate fields, exact-check booleans (`suff`, `min1`, `valid`), and AP/TS metrics.
- `report --strict` enforces schema/metadata validity; `report --official` also requires manifest-based coverage checks.
- `validate` is a dedicated non-aggregation validator, and `migrate-runs` backfills historical run files into strict-compatible v1 records.
