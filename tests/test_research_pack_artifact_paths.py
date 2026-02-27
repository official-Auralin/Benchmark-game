"""
Regression tests for research-pack artifact path integrity.

In the private source repository, these checks enforce that key artifact-path
references in `research_pack/00_index.md` and
`research_pack/55_phase_ah_full_prompt_compliance_audit.md` resolve to
existing files. In the public mirror, private `research_pack/` artifacts are
intentionally absent and content checks are skipped after scope validation.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PACK = ROOT / "research_pack"
INDEX_PATH = RESEARCH_PACK / "00_index.md"
AUDIT_PATH = RESEARCH_PACK / "55_phase_ah_full_prompt_compliance_audit.md"

RECENT_ARTIFACTS_HEADER = "## Recent Execution Artifacts (G14 + Full Compliance)"
ARTIFACT_PATH_RE = re.compile(r"`(research_pack/\d+_[A-Za-z0-9_.-]+\.md)`")

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _extract_recent_artifact_paths(index_text: str) -> list[str]:
    start = index_text.find(RECENT_ARTIFACTS_HEADER)
    if start < 0:
        return []

    section = index_text[start:].splitlines()[1:]
    paths: list[str] = []
    for line in section:
        if line.startswith("## "):
            break
        paths.extend(ARTIFACT_PATH_RE.findall(line))
    return paths


class TestExtractRecentArtifactPathsUnit(unittest.TestCase):
    def test_header_missing_returns_empty(self) -> None:
        text = """
# Some Other Section

- `research_pack/01_valid_artifact.md`
"""
        self.assertEqual(_extract_recent_artifact_paths(text), [])

    def test_header_present_without_artifacts_returns_empty(self) -> None:
        text = f"""
# Top

{RECENT_ARTIFACTS_HEADER}
Here is some descriptive text with no code spans.

- Bullet item without code
"""
        self.assertEqual(_extract_recent_artifact_paths(text), [])

    def test_parsing_stops_at_next_header(self) -> None:
        text = f"""
# Top

{RECENT_ARTIFACTS_HEADER}
- `research_pack/01_first_artifact.md`
- `research_pack/02_second_artifact.md`

## Some Other Section
- `research_pack/03_should_not_be_included.md`
"""
        self.assertEqual(
            _extract_recent_artifact_paths(text),
            [
                "research_pack/01_first_artifact.md",
                "research_pack/02_second_artifact.md",
            ],
        )

    def test_multiple_code_spans_per_line_capture_only_matching_paths(self) -> None:
        text = f"""
# Top

{RECENT_ARTIFACTS_HEADER}
- `research_pack/10_valid_artifact.md` and `not/a/valid_path.txt`
- `research_pack/11_another_valid.md` and `research_pack/12_third_valid.md`
"""
        self.assertEqual(
            _extract_recent_artifact_paths(text),
            [
                "research_pack/10_valid_artifact.md",
                "research_pack/11_another_valid.md",
                "research_pack/12_third_valid.md",
            ],
        )

    def test_ignores_malformed_paths(self) -> None:
        text = f"""
# Top

{RECENT_ARTIFACTS_HEADER}
- `research_pack/not_a_number_prefix.md`
- `research_pack/12_missing_extension`
- `research_pack/13_invalid+chars!.md`
- `different_prefix/14_valid.md`
- `research_pack/15_valid-okay.md`
"""
        self.assertEqual(
            _extract_recent_artifact_paths(text),
            ["research_pack/15_valid-okay.md"],
        )


class TestResearchPackArtifactPaths(unittest.TestCase):
    def test_research_pack_artifact_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            return

        self.assertTrue(INDEX_PATH.exists(), msg=f"missing index: {INDEX_PATH}")
        self.assertTrue(AUDIT_PATH.exists(), msg=f"missing compliance audit: {AUDIT_PATH}")

    @unittest.skipUnless(
        INDEX_PATH.exists() and AUDIT_PATH.exists(),
        "required research_pack artifact files are absent in this repository scope",
    )
    def test_recent_execution_artifact_paths_resolve(self) -> None:
        index_text = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn(
            RECENT_ARTIFACTS_HEADER,
            index_text,
            msg=(
                "index is missing the recent execution artifacts header; "
                "the index structure may have regressed"
            ),
        )
        recent_paths = _extract_recent_artifact_paths(index_text)
        self.assertGreater(
            len(recent_paths),
            0,
            msg="recent execution artifact section should list at least one artifact",
        )
        self.assertEqual(
            len(recent_paths),
            len(set(recent_paths)),
            msg="recent execution artifact list should not contain duplicate paths",
        )

        missing = [path for path in recent_paths if not (ROOT / path).exists()]
        self.assertFalse(
            missing,
            msg=f"recent execution artifact paths missing on disk: {missing}",
        )

    @unittest.skipUnless(
        INDEX_PATH.exists() and AUDIT_PATH.exists(),
        "required research_pack artifact files are absent in this repository scope",
    )
    def test_index_artifact_paths_are_covered_by_compliance_audit(self) -> None:
        index_text = INDEX_PATH.read_text(encoding="utf-8")
        audit_text = AUDIT_PATH.read_text(encoding="utf-8")

        index_paths = _extract_recent_artifact_paths(index_text)
        audit_paths = set(ARTIFACT_PATH_RE.findall(audit_text))
        self.assertGreater(
            len(index_paths),
            0,
            msg="recent execution artifact section should not be empty",
        )

        missing_on_disk = sorted(path for path in audit_paths if not (ROOT / path).exists())
        self.assertFalse(
            missing_on_disk,
            msg=f"compliance audit references missing artifact paths: {missing_on_disk}",
        )

        latest_index_path = index_paths[-1]
        self.assertIn(
            latest_index_path,
            audit_paths,
            msg=(
                "latest artifact in 00_index.md should also be referenced in "
                f"the compliance audit: {latest_index_path}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
