"""
Regression tests for static integrity of ``Spec.tex``.

These checks catch documentation regressions that break self-contained reading
or LaTeX compilation assumptions: duplicate or unresolved labels, unresolved
DEC glossary references, and citation keys not present in the canonical
``sources.bib`` inventory. Content checks run only in private source scope
where the Research Pack bibliography is present.
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
SPEC_PATH = ROOT / "Spec.tex"
RESEARCH_PACK = ROOT / "research_pack"
SOURCES_BIB_PATH = RESEARCH_PACK / "01_sources" / "sources.bib"
PRIVATE_SPEC_CHECK_PATHS = (SPEC_PATH, RESEARCH_PACK, SOURCES_BIB_PATH)

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\ref\{([^}]+)\}")
DEFREF_RE = re.compile(r"\\defref\{([^}]+)\}")
DECTAG_RE = re.compile(r"\\dectag\{(\d{3}(?:[a-z])?)\}")
DECREF_RE = re.compile(r"\\decref\{(\d{3}(?:[a-z])?)\}")
CITE_CMD_RE = re.compile(
    r"\\cite[a-zA-Z*]*"         # \cite, \citep, \citet, etc.
    r"(?:\[[^\]]*\]){0,2}"      # optional natbib arguments
    r"\{([^}]+)\}",
    flags=re.S,
)
BIB_KEY_RE = re.compile(r"@\w+\{([^,]+),")

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _strip_latex_comments(text: str) -> str:
    """Remove unescaped '%' comments so regex checks ignore commented code."""
    stripped_lines = []
    for line in text.splitlines():
        stripped_lines.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(stripped_lines)


class TestSpecTexCommentHelpers(unittest.TestCase):
    def test_strip_latex_comments_behavior(self) -> None:
        src = "foo % this is a comment\nbar % another comment"
        self.assertEqual(_strip_latex_comments(src), "foo \nbar ")

        src_escaped = r"Value is 50\% of total % trailing comment"
        self.assertEqual(
            _strip_latex_comments(src_escaped),
            r"Value is 50\% of total ",
        )

        src_mixed = "prefix % comment\nno_comment_line"
        self.assertEqual(_strip_latex_comments(src_mixed), "prefix \nno_comment_line")


class TestSpecTexIntegrity(unittest.TestCase):
    def test_spec_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            # The public mirror may omit Spec.tex entirely; this test file is
            # mirrored so private-scope logic remains regression-tested there,
            # but Spec.tex content itself is only required in the private source
            # repository unless mirror policy changes.
            return
        self.assertTrue(SPEC_PATH.exists(), msg=f"missing Spec.tex: {SPEC_PATH}")
        self.assertTrue(
            RESEARCH_PACK.exists(),
            msg=f"missing research_pack in private source repo: {RESEARCH_PACK}",
        )
        self.assertTrue(
            SOURCES_BIB_PATH.exists(),
            msg=f"missing sources.bib for Spec.tex citation checks: {SOURCES_BIB_PATH}",
        )

    @unittest.skipUnless(
        all(path.exists() for path in PRIVATE_SPEC_CHECK_PATHS),
        "required private artifacts for Spec.tex integrity checks are absent",
    )
    def test_spec_labels_dec_refs_and_citations_resolve(self) -> None:
        spec_text = _strip_latex_comments(SPEC_PATH.read_text(encoding="utf-8"))
        bib_text = SOURCES_BIB_PATH.read_text(encoding="utf-8")

        labels = LABEL_RE.findall(spec_text)
        label_set = set(labels)
        self.assertEqual(
            len(labels),
            len(label_set),
            msg="Spec.tex contains duplicate \\label{...} definitions",
        )

        refs = {
            ref
            for ref in REF_RE.findall(spec_text) + DEFREF_RE.findall(spec_text)
            if "#" not in ref
        }
        missing_labels = sorted(refs - label_set)
        self.assertFalse(
            missing_labels,
            msg=f"Spec.tex contains unresolved label references: {missing_labels}",
        )

        dec_tags = set(DECTAG_RE.findall(spec_text))
        dec_refs = set(DECREF_RE.findall(spec_text))
        missing_dec_tags = sorted(dec_refs - dec_tags)
        self.assertFalse(
            missing_dec_tags,
            msg=(
                "Spec.tex contains \\decref references without matching "
                f"\\dectag glossary entries: {missing_dec_tags}"
            ),
        )

        cited_keys: set[str] = set()
        for group in CITE_CMD_RE.findall(spec_text):
            for key in group.split(","):
                cleaned = key.strip()
                if cleaned:
                    cited_keys.add(cleaned)

        bib_keys = set(BIB_KEY_RE.findall(bib_text))
        missing_cites = sorted(cited_keys - bib_keys)
        self.assertFalse(
            missing_cites,
            msg=f"Spec.tex cites keys not present in sources.bib: {missing_cites}",
        )


if __name__ == "__main__":
    unittest.main()
