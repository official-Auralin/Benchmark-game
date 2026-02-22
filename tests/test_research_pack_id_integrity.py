"""
Regression tests for research-pack identifier resolvability.

These tests ensure cross-references to SRC/CLM/HYP/PRN/Q/DEC identifiers
resolve to defined entries in the private source repository. In the public
mirror, private research_pack artifacts are intentionally absent and content
checks are skipped after scope validation.
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
SPEC_PATH = ROOT / "Spec.tex"
CLAIMS_LEDGER_PATH = RESEARCH_PACK / "04_claims_ledger.md"
PRINCIPLES_PATH = RESEARCH_PACK / "06_design_principles.md"
OPEN_QUESTIONS_PATH = RESEARCH_PACK / "05_open_questions.md"
SOURCES_CSV_PATH = RESEARCH_PACK / "01_sources" / "sources.csv"
DECISION_LOG_PATH = RESEARCH_PACK / "09_decision_log.md"
RESEARCH_PACK_MARKER_PATHS = (
    RESEARCH_PACK,
    SPEC_PATH,
)
REQUIRED_ID_INTEGRITY_PATHS = (
    CLAIMS_LEDGER_PATH,
    PRINCIPLES_PATH,
    OPEN_QUESTIONS_PATH,
    SOURCES_CSV_PATH,
    DECISION_LOG_PATH,
)
PRIVATE_ID_INTEGRITY_PATHS = RESEARCH_PACK_MARKER_PATHS + REQUIRED_ID_INTEGRITY_PATHS
ID_DEFINITION_PATTERNS = {
    "SRC": re.compile(r"^(SRC-\d{3}),", flags=re.M),
    "CLM": re.compile(r"^(CLM-\d{3})\s+—", flags=re.M),
    "HYP": re.compile(r"^(HYP-\d{3})\s+—", flags=re.M),
    "PRN": re.compile(r"^(PRN-\d{3})\s+—", flags=re.M),
    "Q": re.compile(r"^(Q-\d{3})\s+—", flags=re.M),
    "DEC": re.compile(r"^(DEC-\d{3}(?:[a-z])?)\s+—", flags=re.M),
}
ID_REFERENCE_PATTERNS = {
    "SRC": re.compile(r"SRC-\d{3}"),
    "CLM": re.compile(r"CLM-\d{3}"),
    "HYP": re.compile(r"HYP-\d{3}"),
    "PRN": re.compile(r"PRN-\d{3}"),
    "Q": re.compile(r"Q-\d{3}"),
    "DEC": re.compile(r"DEC-\d{3}[a-z]?"),
}

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


class TestResearchPackIdIntegrity(unittest.TestCase):
    def test_research_pack_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            return

        self.assertTrue(
            RESEARCH_PACK.exists(),
            msg=f"missing research_pack in private source repo: {RESEARCH_PACK}",
        )
        for path in PRIVATE_ID_INTEGRITY_PATHS:
            self.assertTrue(
                path.exists(),
                msg=f"missing required research-pack integrity artifact: {path}",
            )

    @unittest.skipUnless(
        all(path.exists() for path in PRIVATE_ID_INTEGRITY_PATHS),
        "required research_pack artifacts absent in this repository scope",
    )
    def test_identifier_references_resolve(self) -> None:
        claims_text = CLAIMS_LEDGER_PATH.read_text(encoding="utf-8")
        prn_text = PRINCIPLES_PATH.read_text(encoding="utf-8")
        q_text = OPEN_QUESTIONS_PATH.read_text(encoding="utf-8")
        sources_csv_text = SOURCES_CSV_PATH.read_text(encoding="utf-8")
        dec_text = DECISION_LOG_PATH.read_text(encoding="utf-8")

        defined = {
            "SRC": set(ID_DEFINITION_PATTERNS["SRC"].findall(sources_csv_text)),
            "CLM": set(ID_DEFINITION_PATTERNS["CLM"].findall(claims_text)),
            "HYP": set(ID_DEFINITION_PATTERNS["HYP"].findall(claims_text)),
            "PRN": set(ID_DEFINITION_PATTERNS["PRN"].findall(prn_text)),
            "Q": set(ID_DEFINITION_PATTERNS["Q"].findall(q_text)),
            "DEC": set(ID_DEFINITION_PATTERNS["DEC"].findall(dec_text)),
        }

        refs = {kind: set() for kind in ID_REFERENCE_PATTERNS}
        files = [*RESEARCH_PACK.rglob("*.md"), SPEC_PATH]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for kind, pat in ID_REFERENCE_PATTERNS.items():
                refs[kind].update(pat.findall(text))

        for kind in ("SRC", "CLM", "HYP", "PRN", "Q", "DEC"):
            missing = sorted(refs[kind] - defined[kind])
            self.assertFalse(
                missing,
                msg=f"unresolved {kind} identifiers found: {missing}",
            )


if __name__ == "__main__":
    unittest.main()
