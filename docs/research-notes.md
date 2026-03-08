# GF-01 Research Notes

This page is the compact paper-facing rationale for the benchmark. The
underlying evidence chain remains in `research_pack/`.

## Core Thesis

GF-01 is meant to make temporal causal reasoning measurable in a way that is:

- formally specified,
- machine-checkable,
- usable by both humans and agents,
- and defensible enough for benchmark publication.

The human-playable layer matters because it makes the task legible and
experimentally usable beyond purely formal evaluation.

## Working Hypotheses

- Human and agent performance should be comparable under the same canonical
  observation contract, even when their interfaces differ.
- Exact causal-certificate validation is a better backbone for benchmark
  credibility than free-form textual grading.
- A fun, legible human interface increases the benchmark's research reach
  without weakening formal rigor if parity is enforced correctly.

## Key Threats To Validity

The authoritative threat register remains
`research_pack/26_phase_h2_threats_validity.md`. The current highest-risk
classes are:

- hidden-state leakage through renderer affordances,
- shortcut strategies that optimize goal achievement without causal validity,
- public-seed overfitting,
- and reproducibility drift across environments.

## Ablation And Adversarial Priorities

The authoritative ablation plan remains
`research_pack/27_phase_h3_ablations_adversarial_checks.md`. Near-term
priority checks are:

- renderer parity,
- greedy-versus-certified divergence,
- public/private gap auditing,
- deterministic replay,
- and diagnostic-governance calibration.

## Publication Path

For an AAAI-style benchmark paper, the repo should always support:

- a fixed formal specification,
- a stable reference implementation,
- retained pilot and benchmark artifacts,
- documented threats and ablations,
- and a public mirror containing the publishable subset.

The compliance snapshot in
`research_pack/55_phase_ah_full_prompt_compliance_audit.md` should remain a
compact bridge between the canonical docs and the deeper evidence library.
