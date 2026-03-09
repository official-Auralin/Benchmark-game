# Human Data Governance

GF-01 is currently a local research harness. The repository is not a public
data-collection platform, does not claim production security controls, and
should not be described as a deployed human-study system.

## Current Boundary

- The checked-in harness supports local pilot workflows and retained benchmark
  artifacts.
- The repository does not promise hosted account systems, participant identity
  management, encrypted-at-rest storage, or operator-facing deletion workflows.
- Any human-play session artifacts kept in-repo are operator-managed research
  materials, not evidence of a public collection deployment.

## If The Project Moves To Public Collection

Before a public app, website, or remotely hosted study deployment, the project
must define and document:

- consent language and the exact moment consent is recorded,
- whether identifiers are anonymous, pseudonymous, or directly identifying,
- storage location and access controls for raw session data,
- retention periods for raw data and derived artifacts,
- withdrawal/deletion expectations and operational process,
- transport/security assumptions for any hosted submission path,
- separation between public benchmark artifacts and private human-subject data.

## Pre-Deployment Checklist

- Publish a study-specific consent and participant-information document.
- Declare what identifiers are collected and why.
- Document storage locations, retention windows, and who can access them.
- Document deletion/withdrawal handling and any limits on retroactive removal.
- Review whether benchmark artifacts can leak participant information.
- Add explicit security assumptions for transport, authentication, and backups.
- Re-review mirror policy so no human-subject data enters the public GitHub mirror.
