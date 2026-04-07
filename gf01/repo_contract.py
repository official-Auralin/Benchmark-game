"""Shared repository contract and sibling-topology constants."""

from __future__ import annotations

PRIMARY_REPO_REQUIREMENTS_RELATIVE_PATHS = (
    "requirements.txt",
    "requirements-core.txt",
    "requirements-human-ui.txt",
    "requirements-paper-artifact.txt",
    "requirements-dev.txt",
)

PRIMARY_REPO_SPEC_RELATIVE_PATHS = (
    "spec/Spec.pdf",
    "spec/overview.md",
    "spec/contracts.md",
    "spec/environment.md",
    "spec/parity.md",
    "spec/acceptance-tests.md",
    "spec/plan.md",
    "spec/contract_inventory.json",
)

RETAINED_PUBLIC_ARTIFACT_RELATIVE_PATHS = (
    "pilot_freeze/gf01_pilot_freeze_v1",
    "pilot_runs/gf01_pilot_campaign_v1",
)

PRIMARY_REPO_PUBLIC_INCLUDE_PATHS = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    *PRIMARY_REPO_REQUIREMENTS_RELATIVE_PATHS,
    "docs",
    *PRIMARY_REPO_SPEC_RELATIVE_PATHS,
    ".github/workflows/gf01-gate.yml",
    "gf01",
    "tests",
    *RETAINED_PUBLIC_ARTIFACT_RELATIVE_PATHS,
)

REQUIRED_PRIMARY_LAYOUT_RELATIVE_PATHS = (
    "spec/Spec.pdf",
    "docs/LOCAL_COMPANION.md",
    *RETAINED_PUBLIC_ARTIFACT_RELATIVE_PATHS,
)

SPEC_SOURCE_ROOT = "../spec_source"
SPEC_SOURCE_SPEC_TEX = f"{SPEC_SOURCE_ROOT}/Spec.tex"
SPEC_SOURCE_SOURCES_BIB = f"{SPEC_SOURCE_ROOT}/sources.bib"

PRIVATE_COMPANION_ROOT = "../gf01_private_companion"
PRIVATE_COMPANION_SKILL_ROOT = (
    f"{PRIVATE_COMPANION_ROOT}/skill/gf01-private-companion"
)
PRIVATE_COMPANION_SPEC_MD = (
    f"{PRIVATE_COMPANION_SKILL_ROOT}/references/spec/spec.md"
)
PRIVATE_COMPANION_RESEARCH_PACK_ROOT = (
    f"{PRIVATE_COMPANION_ROOT}/source/research_pack"
)
PRIVATE_COMPANION_SOURCES_BIB = (
    f"{PRIVATE_COMPANION_RESEARCH_PACK_ROOT}/01_sources/sources.bib"
)

FORBIDDEN_RUNTIME_LAYOUT_TOKENS = (
    "research_pack/",
    "public_repo/",
    f"{SPEC_SOURCE_ROOT}/",
    f"{PRIVATE_COMPANION_ROOT}/",
)
