"""
Regression tests for the local public-mirror sync policy script.

These tests validate the allowlist/exclude behavior in
``scripts/sync_public_repo.py`` so accidental leakage of private artifacts or
drift in mirrored content policy is caught in the private source repository.
In the public mirror, the script is intentionally absent and content checks are
skipped after scope validation.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT_PATH = ROOT / "scripts" / "sync_public_repo.py"
RESEARCH_PACK_PATH = ROOT / "research_pack"
PUBLIC_REPO_PATH = ROOT / "public_repo"
IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _load_sync_module():
    spec = importlib.util.spec_from_file_location(
        "sync_public_repo_module_for_tests",
        SYNC_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load spec for {SYNC_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSyncPublicRepoPolicy(unittest.TestCase):
    def test_sync_script_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                SYNC_SCRIPT_PATH.exists(),
                msg="public mirror should not include local sync script",
            )
            return
        self.assertTrue(
            SYNC_SCRIPT_PATH.exists(),
            msg=f"missing sync script in source repo: {SYNC_SCRIPT_PATH}",
        )

    @unittest.skipUnless(
        SYNC_SCRIPT_PATH.exists(),
        "sync_public_repo.py is absent in this repository scope",
    )
    def test_public_include_allowlist_policy(self) -> None:
        sync = _load_sync_module()
        include_paths = tuple(sync.PUBLIC_INCLUDE_PATHS)

        self.assertEqual(
            include_paths,
            (
                "README.md",
                "Spec.pdf",
                ".github/workflows/gf01-gate.yml",
                "gf01",
                "tests",
            ),
            msg="PUBLIC_INCLUDE_PATHS changed; review public mirror policy explicitly",
        )
        for forbidden in ("research_pack", "readings", "scripts", ".gitignore"):
            self.assertNotIn(
                forbidden,
                include_paths,
                msg=f"public mirror allowlist must not include {forbidden}",
            )

    @unittest.skipUnless(
        SYNC_SCRIPT_PATH.exists(),
        "sync_public_repo.py is absent in this repository scope",
    )
    def test_ignore_policy_filters_python_cache_artifacts(self) -> None:
        sync = _load_sync_module()
        names = [
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
            "module.pyc",
            "module.pyo",
            "profile.stats",
            "keep.py",
            "keep.txt",
        ]
        ignored = sync.ignore_names(".", names)

        for expected in (
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
            "module.pyc",
            "module.pyo",
            "profile.stats",
        ):
            self.assertIn(expected, ignored)
        for kept in ("keep.py", "keep.txt"):
            self.assertNotIn(kept, ignored)

    @unittest.skipUnless(
        SYNC_SCRIPT_PATH.exists(),
        "sync_public_repo.py is absent in this repository scope",
    )
    def test_copy_public_subset_obeys_allowlist_and_excludes(self) -> None:
        sync = _load_sync_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "src"
            target = tmp_path / "dst"
            root.mkdir()
            target.mkdir()

            # Build only the allowlisted content plus private/noise artifacts.
            (root / "README.md").write_text("readme\n", encoding="utf-8")
            (root / "Spec.pdf").write_bytes(b"%PDF-1.4\n")
            wf = root / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "gf01-gate.yml").write_text("name: test\njobs: {}\n", encoding="utf-8")

            gf01_dir = root / "gf01"
            gf01_dir.mkdir()
            (gf01_dir / "__init__.py").write_text("", encoding="utf-8")
            (gf01_dir / "core.py").write_text("x = 1\n", encoding="utf-8")
            (gf01_dir / "__pycache__").mkdir()
            (gf01_dir / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"x")

            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_ok.py").write_text("def test_ok(): pass\n", encoding="utf-8")
            (tests_dir / ".pytest_cache").mkdir()
            (tests_dir / ".pytest_cache" / "state").write_text("x", encoding="utf-8")

            # Private/local-only paths must never be mirrored.
            rp = root / "research_pack"
            rp.mkdir()
            (rp / "secret.md").write_text("private\n", encoding="utf-8")
            scripts_dir = root / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "sync_public_repo.py").write_text("# local script\n", encoding="utf-8")

            sync.copy_public_subset(root, target)

            # Allowlisted paths are copied.
            self.assertTrue((target / "README.md").exists())
            self.assertTrue((target / "Spec.pdf").exists())
            self.assertTrue((target / ".github" / "workflows" / "gf01-gate.yml").exists())
            self.assertTrue((target / "gf01" / "core.py").exists())
            self.assertTrue((target / "tests" / "test_ok.py").exists())

            # Excluded cache/noise artifacts are not copied.
            self.assertFalse((target / "gf01" / "__pycache__").exists())
            self.assertFalse((target / "tests" / ".pytest_cache").exists())

            # Private/local-only paths are not copied because they are not allowlisted.
            self.assertFalse((target / "research_pack").exists())
            self.assertFalse((target / "scripts").exists())

    @unittest.skipUnless(
        SYNC_SCRIPT_PATH.exists(),
        "sync_public_repo.py is absent in this repository scope",
    )
    def test_clear_target_preserves_git_directory(self) -> None:
        sync = _load_sync_module()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            (target / ".git").mkdir(parents=True)
            (target / ".git" / "config").write_text("[core]\n", encoding="utf-8")
            (target / "old.txt").write_text("stale\n", encoding="utf-8")
            (target / "old_dir").mkdir()
            (target / "old_dir" / "nested.txt").write_text("stale\n", encoding="utf-8")

            sync.clear_target(target)

            self.assertTrue((target / ".git").exists())
            self.assertTrue((target / ".git" / "config").exists())
            self.assertFalse((target / "old.txt").exists())
            self.assertFalse((target / "old_dir").exists())


if __name__ == "__main__":
    unittest.main()
