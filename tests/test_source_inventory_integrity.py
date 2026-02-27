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
import re
import unittest
from pathlib import Path

try:
    from .repo_scope import is_public_mirror
except ImportError:  # pragma: no cover - discover mode imports test modules top-level.
    from repo_scope import is_public_mirror


ROOT = Path(__file__).resolve().parents[1]
SOURCES_CSV_PATH = ROOT / "research_pack" / "01_sources" / "sources.csv"
SOURCES_BIB_PATH = ROOT / "research_pack" / "01_sources" / "sources.bib"
SOURCE_NOTES_DIR = ROOT / "research_pack" / "03_notes"

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)

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


def _bib_entries_by_key(bib_text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for chunk in re.split(r"\n(?=@\w+\{)", bib_text.strip()):
        match = re.match(r"@\w+\{([^,]+),", chunk)
        if match is not None:
            entries[match.group(1)] = chunk
    return entries


def _bib_field_value(entry_text: str, field_name: str) -> str:
    field_pattern = rf"{re.escape(field_name)}\s*=\s*(\{{[^}}]*\}}|\"[^\"]*\")"
    match = re.search(field_pattern, entry_text, flags=re.IGNORECASE)
    if match is None:
        return ""
    value = match.group(1).strip()
    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1].strip()
    return value


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


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
        bib_entries = _bib_entries_by_key(bib_text)
        csv_citation_keys = set(citation_keys)
        csv_source_ids = set(source_ids)
        self.assertTrue(
            bib_keys.issubset(csv_citation_keys),
            msg=(
                "Every citation key in sources.bib must exist as a "
                "citation_key in sources.csv"
            ),
        )
        self.assertTrue(
            bib_source_ids.issubset(csv_source_ids),
            msg=(
                "Every source_id referenced in sources.bib must exist as a "
                "source_id in sources.csv"
            ),
        )

        required_statuses = {"skimming", "deep-read", "extracted"}
        for row in rows:
            source_id = row["source_id"]
            citation_key = row["citation_key"]
            landing_url = row["publisher_landing_url"].strip()
            pdf_url = row["pdf_url"].strip()
            rights_notes = row["license/rights_notes"].lower()
            status = row["status"].strip().lower()
            doi = row["doi"].strip().lower()
            arxiv_id = row["arxiv_id"].strip()
            bib_entry = bib_entries.get(citation_key, "")

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
            self.assertTrue(
                bib_entry,
                msg=f"{source_id} missing BibTeX entry for citation_key={citation_key}",
            )

            normalized_landing = _normalize_url(landing_url)
            bib_urls = {
                _normalize_url(url)
                for url in re.findall(r"https?://[^}\s,]+", bib_entry)
            }
            self.assertIn(
                normalized_landing,
                bib_urls,
                msg=(
                    f"{source_id} publisher_landing_url must match a URL in "
                    "its BibTeX entry"
                ),
            )

            if doi:
                bib_doi = _bib_field_value(bib_entry, "doi").lower()
                self.assertEqual(
                    bib_doi,
                    doi,
                    msg=f"{source_id} DOI mismatch between sources.csv and sources.bib",
                )
                doi_suffix = doi.split("/", 1)[1] if "/" in doi else doi
                doi_tail = doi_suffix.split("/")[-1]
                landing_lower = landing_url.lower()
                doi_tokens = [
                    token
                    for token in re.split(r"[^a-z0-9]+", doi)
                    if len(token) >= 5
                ]
                landing_matches_expected_id = (
                    doi in landing_lower
                    or doi_suffix in landing_lower
                    or doi_tail in landing_lower
                    or any(token in landing_lower for token in doi_tokens)
                    or bool(arxiv_id and arxiv_id in landing_lower)
                )
                self.assertTrue(
                    landing_matches_expected_id,
                    msg=(
                        f"{source_id} publisher_landing_url should include DOI "
                        f"or canonical identifier derived from DOI/arXiv metadata "
                        f"(doi={doi}, arxiv_id={arxiv_id or 'N/A'})"
                    ),
                )

            if arxiv_id:
                bib_eprint = _bib_field_value(bib_entry, "eprint")
                self.assertEqual(
                    bib_eprint,
                    arxiv_id,
                    msg=f"{source_id} arXiv ID mismatch between sources.csv and sources.bib",
                )
                self.assertIn(
                    arxiv_id,
                    bib_entry,
                    msg=f"{source_id} BibTeX entry should include arXiv ID {arxiv_id}",
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
