"""Regression tests for docs/spec scaffolding and public contract rules."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from gf01 import meta


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    ROOT / "LICENSE",
    ROOT / "pyproject.toml",
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "CONTRIBUTING.md",
    ROOT / "docs" / "HUMAN_DATA_GOVERNANCE.md",
    ROOT / "docs" / "LOCAL_COMPANION.md",
    ROOT / "docs" / "STYLE.md",
    ROOT / "docs" / "benchmarking.md",
    ROOT / "docs" / "research-notes.md",
    ROOT / "docs" / "ARCHIVE_LOG.md",
    ROOT / "spec" / "overview.md",
    ROOT / "spec" / "contracts.md",
    ROOT / "spec" / "environment.md",
    ROOT / "spec" / "parity.md",
    ROOT / "spec" / "acceptance-tests.md",
    ROOT / "spec" / "plan.md",
    ROOT / "spec" / "contract_inventory.json",
    ROOT / "spec" / "Spec.pdf",
    ROOT / "requirements.txt",
    ROOT / "requirements-core.txt",
    ROOT / "requirements-human-ui.txt",
    ROOT / "requirements-paper-artifact.txt",
    ROOT / "requirements-dev.txt",
)
CONTRACT_INVENTORY_PATH = ROOT / "spec" / "contract_inventory.json"


class TestDocsSpecSync(unittest.TestCase):
    def test_required_docs_and_spec_paths_exist(self) -> None:
        for path in REQUIRED_PATHS:
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

    def test_contract_inventory_lists_retained_public_artifacts(self) -> None:
        payload = json.loads(CONTRACT_INVENTORY_PATH.read_text(encoding="utf-8"))
        include_paths = payload.get("public_include_paths", [])
        self.assertIn("pilot_freeze/gf01_pilot_freeze_v1", include_paths)
        self.assertIn("pilot_runs/gf01_pilot_campaign_v1", include_paths)

    def test_primary_repo_dependency_rule_is_documented(self) -> None:
        req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        human_ui_text = (ROOT / "requirements-human-ui.txt").read_text(
            encoding="utf-8"
        ).lower()
        contrib_text = (ROOT / "docs" / "CONTRIBUTING.md").read_text(
            encoding="utf-8"
        ).lower()
        self.assertIn("primary-repo runtime dependencies only", req_text)
        self.assertIn("pygame-ce==", human_ui_text)
        self.assertIn("do not add local-only", contrib_text)
        self.assertIn("primary repo runtime surface", contrib_text)


if __name__ == "__main__":
    unittest.main()
