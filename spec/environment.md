# GF-01 Environment Contract

This page is the compact operational description of the benchmark environment.
The formal normative source remains `spec/tex_files/Spec.tex`.

## Instance Model

Each GF-01 instance combines:

- a reactive-system object,
- disjoint input and output proposition sets,
- a finite base trace,
- a target effect and target timestep,
- a scoring mode (`normal` or `hard`),
- and an official timestep intervention budget.

Official identity is the normalized formal content of the instance. The
environment is deterministic for fixed normalized content, normalization
version, and benchmark code version. Legacy generator `seed` values may still
appear in artifacts as development provenance, but they are not part of the
official semantic identity.

## Action Model

- At each timestep, the agent or human chooses a partial assignment over the
  current input propositions.
- Inputs may be set to `0`, set to `1`, or left unchanged for that step.
- There are no explicit query actions.
- Official semantics are governed by certificate validity, not reward shaping.
- Legal per-step inputs come from the normalized input-total formal system, not
  from a separate atom-budget rule.

## Observation Model

Canonical observation exposes only benchmark-approved information:

- current timestep,
- currently observed output propositions,
- effect status,
- remaining official timestep budget,
- intervention history in a lossless form,
- submitted and committed step actions when the runtime exposes them,
- mode, target timestep, and other mission metadata needed for play.

Hidden rollout state must never be exposed.
In particular, the base trace remains hidden.

## Objective Semantics

- `hard` mode scores the exact target timestep.
- `normal` mode scores within the allowed trailing window ending at the target
  timestep.
- A submitted certificate is valid only if it is sufficient and singleton
  removal minimal.

## Budgets And Constraints

- Timestep budgets are part of the public contract.
- Optional atom-count metadata may exist for legacy/dev workflows, but it is
  not part of official validity in the active runtime path.
- Action entry, rendering, and reporting may change presentation, but not the
  underlying intervention semantics.

## Episode End

Episodes terminate when the trace horizon is exhausted or when the command
workflow explicitly finishes. Validation and reporting operate on the emitted
artifact contract rather than on UI state.
