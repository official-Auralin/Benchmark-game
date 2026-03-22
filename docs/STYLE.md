# Documentation Style

Keep the repo's active working set small and high signal.

## Rules

- Extend an existing canonical doc before creating a new top-level doc.
- Keep one source of truth for each topic:
  - contracts in `spec/contracts.md`
  - environment semantics in `spec/environment.md`
  - parity rules in `spec/parity.md`
  - active planning in `spec/plan.md`
  - architecture in `docs/ARCHITECTURE.md`
  - benchmarking protocol in `docs/benchmarking.md`
- put durable decisions in
  `../gf01_private_companion/source/research_pack/09_decision_log.md`, not ad
  hoc notes
- keep formal spec changes synchronized across `../spec_source/Spec.tex`,
  `spec/Spec.pdf`, and the companion skill spec reference
- Do not keep chat transcripts, one-off brainstorming dumps, or repetitive
  execution logs in the repo root.
- Prefer short sections, explicit headings, and link-out summaries over long
  chronological narratives.

## Archive Procedure

- If a doc is not part of the canonical set and is not required for scientific
  defensibility, move it to `../archive/Research/<dated-batch>/`.
- Organize archive batches by purpose, not by one flat file dump.
- Add a short note to `docs/ARCHIVE_LOG.md` whenever a new archive batch is
  created.
