# GF-01 Acceptance Scenarios

These scenarios define the highest-value behavioral checks that must stay true
through refactors.

## Core Workflow

- Generate a deterministic instance bundle and split manifest.
- Evaluate a baseline agent against a bundle and emit strict run rows.
- Report and validate runs against a manifest in strict or official mode.
- Migrate legacy runs forward to `gf01.run_record.v1` without changing public
  validation semantics.

## Play Contract

- `gf01 play --agent greedy --renderer-track json` returns a structured,
  machine-checkable payload.
- Track policy, renderer policy, and adaptation policy violations fail with a
  JSON error payload and non-zero exit code.
- Visual play supports `text` and `pygame` backends without changing canonical
  scoring semantics.

## Governance And Release

- Split-policy and release-governance checks preserve their current policy
  semantics.
- Release-report and release-candidate checks keep their current stage-level
  outcomes and validation behavior.
- Public mirror sync continues to copy only the explicit allowlist.

## Pilot And P0

- Pilot freeze/campaign/analyze workflows preserve current artifact contracts.
- P0 template, session, feedback, gate, and init commands preserve their
  machine-checkable summaries and failure modes.
