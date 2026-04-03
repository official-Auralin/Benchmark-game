"""Regression tests for the simplified primary-repo layout."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from gf01 import repo_contract


ROOT = Path(__file__).resolve().parents[1]
RETAINED_PUBLIC_PATHS = (
    *(ROOT / relative_path for relative_path in repo_contract.REQUIRED_PRIMARY_LAYOUT_RELATIVE_PATHS),
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
RUNTIME_LAYOUT_SCAN_EXCLUSIONS = {
    ROOT / "gf01" / "repo_contract.py",
}

PATHLIKE_METHOD_NAMES = frozenset({"joinpath", "glob", "rglob", "open"})
PATHLIKE_RECEIVER_SUFFIXES = ("path", "root", "dir")


def _is_path_constructor(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Path"
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pathlib"
        and node.attr == "Path"
    )


def _is_path_receiver(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        token = node.id.lower()
        return node.id == "ROOT" or token.endswith(PATHLIKE_RECEIVER_SUFFIXES)
    if isinstance(node, ast.Attribute):
        return node.attr.lower().endswith(PATHLIKE_RECEIVER_SUFFIXES) or _is_path_receiver(
            node.value
        )
    if isinstance(node, ast.Call):
        return _is_pathlike_call(node)
    return False


def _is_pathlike_call(node: ast.Call) -> bool:
    func = node.func
    if _is_path_constructor(func):
        return True
    if isinstance(func, ast.Name):
        return func.id == "open"
    return (
        isinstance(func, ast.Attribute)
        and func.attr in PATHLIKE_METHOD_NAMES
        and _is_path_receiver(func.value)
    )


def _string_fragments(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        fragments: list[str] = []
        for value in node.values:
            fragments.extend(_string_fragments(value))
        return fragments

    fragments: list[str] = []
    for child in ast.iter_child_nodes(node):
        fragments.extend(_string_fragments(child))
    return fragments


def _runtime_path_strings(tree: ast.AST) -> list[str]:
    strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            strings.extend(_string_fragments(getattr(node, "value", None)))
            continue
        if isinstance(node, ast.Call) and _is_pathlike_call(node):
            for arg in node.args:
                strings.extend(_string_fragments(arg))
            for keyword in node.keywords:
                strings.extend(_string_fragments(keyword.value))
    return strings


class TestRepoLayoutPolicy(unittest.TestCase):
    def test_runtime_path_string_extractor_ignores_unrelated_open_methods(self) -> None:
        tree = ast.parse(
            'client.open("../spec_source/Spec.tex")\n'
            'pathlib.Path("../spec_source/Spec.tex")\n'
        )
        self.assertEqual(
            _runtime_path_strings(tree),
            ["../spec_source/Spec.tex"],
        )

    def test_retained_public_paths_exist(self) -> None:
        for path in RETAINED_PUBLIC_PATHS:
            self.assertTrue(path.exists(), msg=f"missing retained public path: {path}")

    def test_removed_private_and_mirror_paths_are_absent(self) -> None:
        for path in FORBIDDEN_PATHS + FORBIDDEN_PUBLIC_ARTIFACTS:
            self.assertFalse(path.exists(), msg=f"forbidden path still present: {path}")

    def test_runtime_python_files_do_not_reference_private_layout_paths(self) -> None:
        for path in ROOT.joinpath("gf01").rglob("*.py"):
            if path in RUNTIME_LAYOUT_SCAN_EXCLUSIONS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for fragment in _runtime_path_strings(tree):
                for token in repo_contract.FORBIDDEN_RUNTIME_LAYOUT_TOKENS:
                    self.assertNotIn(
                        token,
                        fragment,
                        msg=(
                            f"runtime file {path} still references forbidden "
                            f"token {token!r} in executable string literal {fragment!r}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
