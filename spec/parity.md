# GF-01 Human And Agent Parity

GF-01 must remain playable by humans and agents without changing the canonical
information contract.

## Parity Rule

The allowed information set is:

`I(s) = O(s) + static mission metadata`

where `O(s)` is the canonical benchmark observation and static mission metadata
includes only benchmark-approved context such as AP names, target timing,
mode, and budget information.

## Allowed Interface Differences

- Text, JSON, and the canonical `GF-01-R1` pygame renderer may reformat the
  same information.
- Visual overlays may summarize or highlight observed information.
- Human-facing mockups may improve readability and onboarding.

## Forbidden Differences

- No renderer may reveal hidden internal state.
- No overlay may encode future trace values or oracle-only statistics.
- UI convenience features must not change scoring semantics or certificate
  validity.

## Validation Expectations

- Renderer parity is enforced by tests and by the ablation plan.
- The canonical `GF-01-R1` pygame backend must provide explicit access to the
  canonical observation payload for parity auditing.
- Derived summaries must be computable only from already observed outputs,
  action history, and static mission metadata.

## Design Artifacts

Historical mockups have been archived under the 2026-03-07 repo-curation batch
because they are not part of the active steering set. The runtime and verifier
remain the source of truth for behavior.
