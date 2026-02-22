"""
Regression tests for research-pack identifier resolvability.

These tests ensure cross-references to SRC/CLM/HYP/PRN/Q/DEC identifiers
resolve to defined entries in the private source repository. In the public
mirror, private research_pack artifacts are intentionally absent and content
checks are skipped after scope validation.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "License Name"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

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
        self.assertTrue(SPEC_PATH.exists(), msg=f"missing Spec.tex: {SPEC_PATH}")

    @unittest.skipUnless(
        RESEARCH_PACK.exists() and SPEC_PATH.exists(),
        "research_pack/Spec.tex absent in this repository scope",
    )
    def test_identifier_references_resolve(self) -> None:
        claims_text = CLAIMS_LEDGER_PATH.read_text(encoding="utf-8")
        prn_text = PRINCIPLES_PATH.read_text(encoding="utf-8")
        q_text = OPEN_QUESTIONS_PATH.read_text(encoding="utf-8")
        sources_csv_text = SOURCES_CSV_PATH.read_text(encoding="utf-8")
        dec_text = DECISION_LOG_PATH.read_text(encoding="utf-8")

        defined = {
            "SRC": set(re.findall(r"^(SRC-\d{3}),", sources_csv_text, flags=re.M)),
            "CLM": set(re.findall(r"^(CLM-\d{3})\s+—", claims_text, flags=re.M)),
            "HYP": set(re.findall(r"^(HYP-\d{3})\s+—", claims_text, flags=re.M)),
            "PRN": set(re.findall(r"^(PRN-\d{3})\s+—", prn_text, flags=re.M)),
            "Q": set(re.findall(r"^(Q-\d{3})\s+—", q_text, flags=re.M)),
            "DEC": set(re.findall(r"^(DEC-\d{3}(?:[a-z])?)\s+—", dec_text, flags=re.M)),
        }

        patterns = {
            "SRC": re.compile(r"SRC-\d{3}"),
            "CLM": re.compile(r"CLM-\d{3}"),
            "HYP": re.compile(r"HYP-\d{3}"),
            "PRN": re.compile(r"PRN-\d{3}"),
            "Q": re.compile(r"Q-\d{3}"),
            "DEC": re.compile(r"DEC-\d{3}[a-z]?"),
        }

        refs = {kind: set() for kind in patterns}
        files = [*RESEARCH_PACK.rglob("*.md"), SPEC_PATH]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for kind, pat in patterns.items():
                refs[kind].update(pat.findall(text))

        for kind in ("SRC", "CLM", "HYP", "PRN", "Q", "DEC"):
            missing = sorted(refs[kind] - defined[kind])
            self.assertFalse(
                missing,
                msg=f"unresolved {kind} identifiers found: {missing}",
            )


if __name__ == "__main__":
    unittest.main()
