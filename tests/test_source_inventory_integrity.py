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
from urllib.parse import parse_qsl, unquote, urlsplit

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


_BIB_ENTRY_START_RE = re.compile(r"(?m)^\s*@\w+\s*\{\s*([^,\s][^,]*?)\s*,")
_URL_RE = re.compile(r"https?://[^\s}]+", flags=re.IGNORECASE)
_TRIVIAL_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def _consume_braced_value(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        return ("", start)
    depth = 0
    idx = start
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Return content without outer braces.
                return (text[start + 1 : idx], idx + 1)
        idx += 1
    return ("", start)


def _consume_quoted_value(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != '"':
        return ("", start)
    idx = start + 1
    escaped = False
    chunks: list[str] = []
    while idx < len(text):
        ch = text[idx]
        if escaped:
            chunks.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return ("".join(chunks), idx + 1)
        else:
            chunks.append(ch)
        idx += 1
    return ("", start)


def _bib_entries_by_key(bib_text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in _BIB_ENTRY_START_RE.finditer(bib_text):
        key = match.group(1).strip()
        open_idx = bib_text.find("{", match.start())
        if open_idx < 0:
            continue
        _, end_idx = _consume_braced_value(bib_text, open_idx)
        if end_idx <= open_idx:
            continue
        entries[key] = bib_text[match.start() : end_idx]
    return entries


def _bib_field_value(entry_text: str, field_name: str) -> str:
    field_pattern = re.compile(
        rf"(?i)(?<![A-Za-z0-9_-]){re.escape(field_name)}\s*=\s*"
    )
    match = field_pattern.search(entry_text)
    if match is None:
        return ""
    idx = match.end()
    while idx < len(entry_text) and entry_text[idx].isspace():
        idx += 1
    if idx >= len(entry_text):
        return ""

    if entry_text[idx] == "{":
        value, _ = _consume_braced_value(entry_text, idx)
        return value.strip()
    if entry_text[idx] == '"':
        value, _ = _consume_quoted_value(entry_text, idx)
        return value.strip()

    # Fallback for bare values (e.g., month = jan).
    end_idx = idx
    while end_idx < len(entry_text) and entry_text[end_idx] not in {",", "\n", "}"}:
        end_idx += 1
    return entry_text[idx:end_idx].strip()


def _normalize_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        return ""

    parts = urlsplit(stripped)
    netloc = parts.netloc.lower()
    path = unquote(parts.path).rstrip("/")
    if path == "/":
        path = ""

    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key.startswith("utm_") or lower_key in _TRIVIAL_QUERY_KEYS:
            continue
        query_pairs.append((lower_key, value))
    query_pairs.sort()

    canonical_query = "&".join(
        f"{key}={value}" if value != "" else key for key, value in query_pairs
    )
    base = f"{netloc}{path}"
    if canonical_query:
        return f"{base}?{canonical_query}"
    return base


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        urls.append(match.group(0).rstrip(".,);]"))
    return urls


class TestSourceInventoryHelperParsing(unittest.TestCase):
    def test_bib_entries_parser_tolerates_whitespace_comments_and_nested_braces(self) -> None:
        bib_text = """
% comment line
   @article{KeyA,
      title = {One},
      note = {source_id: SRC-001}
   }

@misc{KeyB,
  title = {Two},
  howpublished = {\\url{https://example.com/path?q=1}},
  note = {source_id: SRC-002}
}
"""
        entries = _bib_entries_by_key(bib_text)
        self.assertEqual(set(entries), {"KeyA", "KeyB"})
        self.assertIn("source_id: SRC-001", entries["KeyA"])
        self.assertIn("source_id: SRC-002", entries["KeyB"])

    def test_bib_field_value_handles_nested_braces_and_multiline_values(self) -> None:
        entry = """
@article{KeyC,
  title = {A {Nested} Title},
  note = {source_id: SRC-123},
  doi = "10.1000/xyz"
}
"""
        self.assertEqual(_bib_field_value(entry, "title"), "A {Nested} Title")
        self.assertEqual(_bib_field_value(entry, "note"), "source_id: SRC-123")
        self.assertEqual(_bib_field_value(entry, "doi"), "10.1000/xyz")

    def test_normalize_url_ignores_scheme_fragment_and_trivial_query(self) -> None:
        url_a = "http://Example.com/path/?utm_source=x&ref=abc&id=7#frag"
        url_b = "https://example.com/path?id=7"
        self.assertEqual(_normalize_url(url_a), _normalize_url(url_b))


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
        bib_entries = _bib_entries_by_key(bib_text)
        bib_keys = set(bib_entries)
        bib_source_ids = set()
        for entry in bib_entries.values():
            note_text = _bib_field_value(entry, "note")
            note_match = re.search(r"source_id\s*:\s*(SRC-\d{3})", note_text)
            if note_match is not None:
                bib_source_ids.add(note_match.group(1))
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
                for url in _extract_urls(bib_entry)
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
