"""
Regression tests for Research Pack source-inventory metadata integrity.

These tests enforce internal consistency across sources.csv, sources.bib,
and source notes in the private source repository. In the public mirror,
the private research_pack is intentionally absent, so integrity content
checks are skipped by design after scope validation.
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

import csv
import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SPEC_PATH = ROOT / "Spec.tex"
SOURCES_CSV_PATH = ROOT / "research_pack" / "01_sources" / "sources.csv"
SOURCES_BIB_PATH = ROOT / "research_pack" / "01_sources" / "sources.bib"
SOURCE_NOTES_DIR = ROOT / "research_pack" / "03_notes"

_repo_scope = os.environ.get("GF01_REPO_SCOPE", "").strip().lower()
if _repo_scope in {"public", "public_mirror"}:
    IS_PUBLIC_MIRROR = True
elif _repo_scope in {"private", "source"}:
    IS_PUBLIC_MIRROR = False
else:
    # Fallback repo detection for local runs:
    # private source repo includes Spec.tex, public mirror does not.
    IS_PUBLIC_MIRROR = not PRIVATE_SPEC_PATH.exists()

_MISSING_PDF_JUSTIFICATION_KEYWORDS = (
    "no public pdf",
    "repository source",
    "web documentation",
    "web post",
    "paywalled",
    "subscription",
    "pdf unavailable",
    "not applicable",
)


class TestSourceInventoryIntegrity(unittest.TestCase):
    def test_source_inventory_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                SOURCES_CSV_PATH.exists(),
                msg=(
                    "public mirror should not include private source inventory: "
                    f"{SOURCES_CSV_PATH}"
                ),
            )
            self.assertFalse(
                SOURCES_BIB_PATH.exists(),
                msg=(
                    "public mirror should not include private source inventory: "
                    f"{SOURCES_BIB_PATH}"
                ),
            )
            return

        self.assertTrue(
            SOURCES_CSV_PATH.exists(),
            msg=f"missing source inventory CSV in private repo: {SOURCES_CSV_PATH}",
        )
        self.assertTrue(
            SOURCES_BIB_PATH.exists(),
            msg=f"missing source inventory BibTeX in private repo: {SOURCES_BIB_PATH}",
        )

    @unittest.skipUnless(
        SOURCES_CSV_PATH.exists() and SOURCES_BIB_PATH.exists(),
        "source inventory files are absent in this repository scope",
    )
    def test_source_inventory_csv_bib_and_notes_integrity(self) -> None:
        with SOURCES_CSV_PATH.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertGreater(len(rows), 0, msg="sources.csv must include at least one row")

        source_ids = [row["source_id"] for row in rows]
        citation_keys = [row["citation_key"] for row in rows]
        self.assertEqual(
            len(source_ids),
            len(set(source_ids)),
            msg="source_id values in sources.csv must be unique",
        )
        self.assertEqual(
            len(citation_keys),
            len(set(citation_keys)),
            msg="citation_key values in sources.csv must be unique",
        )

        bib_text = SOURCES_BIB_PATH.read_text(encoding="utf-8")
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
        bib_source_ids = set(
            re.findall(
                r"note\s*=\s*\{[^}]*source_id\s*:\s*(SRC-\d{3})[^}]*\}",
                bib_text,
            )
        )

        required_statuses = {"skimming", "deep-read", "extracted"}
        for row in rows:
            source_id = row["source_id"]
            citation_key = row["citation_key"]
            landing_url = row["publisher_landing_url"].strip()
            pdf_url = row["pdf_url"].strip()
            rights_notes = row["license/rights_notes"].lower()
            status = row["status"].strip().lower()

            self.assertTrue(
                landing_url,
                msg=f"{source_id} missing publisher_landing_url",
            )
            self.assertIn(
                citation_key,
                bib_keys,
                msg=f"{source_id} citation_key missing in sources.bib: {citation_key}",
            )
            self.assertIn(
                source_id,
                bib_source_ids,
                msg=f"{source_id} missing note={{source_id: ...}} mapping in sources.bib",
            )

            if not pdf_url or pdf_url.upper() == "N/A":
                self.assertTrue(
                    any(k in rights_notes for k in _MISSING_PDF_JUSTIFICATION_KEYWORDS),
                    msg=(
                        f"{source_id} missing explicit missing-PDF justification in "
                        "license/rights_notes"
                    ),
                )

            if status in required_statuses:
                note_path = SOURCE_NOTES_DIR / f"{source_id}_notes.md"
                self.assertTrue(
                    note_path.exists(),
                    msg=f"{source_id} status={status} requires notes file: {note_path}",
                )


if __name__ == "__main__":
    unittest.main()
