"""Regression tests for the simplified primary-repo layout."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETAINED_PUBLIC_PATHS = (
    ROOT / "spec" / "Spec.pdf",
    ROOT / "pilot_freeze" / "gf01_pilot_freeze_v1",
    ROOT / "pilot_runs" / "gf01_pilot_campaign_v1",
    ROOT / "docs" / "LOCAL_COMPANION.md",
)
FORBIDDEN_PATHS = (
    ROOT / "public_repo",
    ROOT / "research_pack",
    ROOT / "spec" / "tex_files",
    ROOT / "scripts" / "build_spec.py",
    ROOT / "scripts" / "sync_public_repo.py",
    ROOT / "docs" / "PUBLIC_MIRROR.md",
    ROOT / "tests" / "repo_scope.py",
    ROOT / "tests" / "test_sync_public_repo_policy.py",
    ROOT / "tests" / "test_research_pack_id_integrity.py",
    ROOT / "tests" / "test_research_pack_artifact_paths.py",
    ROOT / "tests" / "test_source_inventory_integrity.py",
    ROOT / "tests" / "test_compliance_audit_consistency.py",
    ROOT / "tests" / "test_decision_log_integrity.py",
    ROOT / "tests" / "test_decision_log_structure.py",
    ROOT / "tests" / "test_spec_tex_integrity.py",
)
FORBIDDEN_PUBLIC_ARTIFACTS = (
    ROOT / "pilot_freeze" / "hyp018_matched_mode_v4_n240",
    ROOT / "pilot_runs" / "hyp018_matched_mode_v4_n240",
    ROOT / "pilot_runs" / "hyp018_trend_summary",
)
FORBIDDEN_RUNTIME_TOKENS = (
    "research_pack/",
    "public_repo/",
    "../spec_source/",
    "../gf01_private_companion/",
)


class TestRepoLayoutPolicy(unittest.TestCase):
    def test_retained_public_paths_exist(self) -> None:
        for path in RETAINED_PUBLIC_PATHS:
            self.assertTrue(path.exists(), msg=f"missing retained public path: {path}")

    def test_removed_private_and_mirror_paths_are_absent(self) -> None:
        for path in FORBIDDEN_PATHS + FORBIDDEN_PUBLIC_ARTIFACTS:
            self.assertFalse(path.exists(), msg=f"forbidden path still present: {path}")

    def test_runtime_python_files_do_not_reference_private_layout_paths(self) -> None:
        for path in ROOT.joinpath("gf01").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_RUNTIME_TOKENS:
                self.assertNotIn(
                    token,
                    text,
                    msg=f"runtime file {path} still references forbidden token {token!r}",
                )


if __name__ == "__main__":
    unittest.main()
