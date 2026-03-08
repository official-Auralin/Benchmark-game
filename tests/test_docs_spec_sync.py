"""Regression tests for docs/spec scaffolding and public-scope dependency rules."""

from __future__ import annotations

import unittest
from pathlib import Path

from gf01 import meta

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
COMMON_REQUIRED_PATHS = (
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "CONTRIBUTING.md",
    ROOT / "docs" / "STYLE.md",
    ROOT / "docs" / "benchmarking.md",
    ROOT / "docs" / "research-notes.md",
    ROOT / "docs" / "PUBLIC_MIRROR.md",
    ROOT / "docs" / "ARCHIVE_LOG.md",
    ROOT / "spec" / "overview.md",
    ROOT / "spec" / "contracts.md",
    ROOT / "spec" / "environment.md",
    ROOT / "spec" / "parity.md",
    ROOT / "spec" / "acceptance-tests.md",
    ROOT / "spec" / "plan.md",
    ROOT / "spec" / "Spec.pdf",
    ROOT / "requirements.txt",
)
SOURCE_ONLY_REQUIRED_PATHS = (
    ROOT / "spec" / "tex_files" / "Spec.tex",
)
IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


class TestDocsSpecSync(unittest.TestCase):
    def test_required_docs_and_spec_paths_exist(self) -> None:
        required_paths = list(COMMON_REQUIRED_PATHS)
        if not IS_PUBLIC_MIRROR:
            required_paths.extend(SOURCE_ONLY_REQUIRED_PATHS)
        for path in required_paths:
            self.assertTrue(path.exists(), msg=f"missing required artifact: {path}")

    def test_contracts_page_mentions_current_schema_and_policy_versions(self) -> None:
        text = (ROOT / "spec" / "contracts.md").read_text(encoding="utf-8")
        required_tokens = (
            meta.INSTANCE_BUNDLE_SCHEMA_VERSION,
            meta.RUN_RECORD_SCHEMA_VERSION,
            meta.SPLIT_MANIFEST_SCHEMA_VERSION,
            meta.PILOT_FREEZE_SCHEMA_VERSION,
            meta.ADAPTATION_POLICY_VERSION,
            meta.RENDERER_POLICY_VERSION,
            meta.IDENTIFIABILITY_POLICY_VERSION,
            meta.COMPLEXITY_POLICY_VERSION,
            meta.BASELINE_PANEL_POLICY_VERSION,
            meta.SPLIT_POLICY_VERSION,
            meta.ROTATION_POLICY_VERSION,
            meta.TOOL_POLICY_VERSION,
        )
        for token in required_tokens:
            self.assertIn(token, text, msg=f"missing contract token: {token}")

    def test_public_scope_dependency_rule_is_documented(self) -> None:
        req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        contrib_text = (ROOT / "docs" / "CONTRIBUTING.md").read_text(
            encoding="utf-8"
        ).lower()
        self.assertIn("mirrored repository only", req_text)
        self.assertIn("do not add local-only", contrib_text)
        self.assertIn("mirrored/public files", contrib_text)


if __name__ == "__main__":
    unittest.main()
