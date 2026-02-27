"""
Regression tests for decision-log entry structure.

In the private source repository, these checks enforce that each DEC entry in
`research_pack/09_decision_log.md` includes the required canonical fields in
order. In the public mirror, private research-pack artifacts are intentionally
absent and content checks are skipped after scope validation.
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
DECISION_LOG_PATH = RESEARCH_PACK / "09_decision_log.md"

DEC_ENTRY_RE = re.compile(r"^(DEC-\d{3}[a-z]?)\s+—", flags=re.M)
DEC_FIELD_RE = {
    "decision_statement": re.compile(r"^- Decision statement:\s+", flags=re.M),
    "chosen_option": re.compile(r"^- Chosen option:\s+", flags=re.M),
    "rationale_cited": re.compile(r"^- Rationale \(cited\):\s+", flags=re.M),
    "open_risks": re.compile(r"^- Open risks:\s+", flags=re.M),
}

IS_PUBLIC_MIRROR = is_public_mirror(ROOT)


def _decision_entries(text: str) -> list[tuple[str, str]]:
    matches = list(DEC_ENTRY_RE.finditer(text))
    entries: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        entries.append((match.group(1), text[start:end]))
    return entries


class TestDecisionLogStructure(unittest.TestCase):
    def test_decision_log_presence_by_repo_scope(self) -> None:
        if IS_PUBLIC_MIRROR:
            self.assertFalse(
                RESEARCH_PACK.exists(),
                msg="public mirror should not include private research_pack/",
            )
            return

        self.assertTrue(
            DECISION_LOG_PATH.exists(),
            msg=f"missing decision log in source repo: {DECISION_LOG_PATH}",
        )

    @unittest.skipUnless(
        DECISION_LOG_PATH.exists(),
        "decision log is absent in this repository scope",
    )
    def test_decision_entries_contain_required_fields_in_order(self) -> None:
        text = DECISION_LOG_PATH.read_text(encoding="utf-8")
        entries = _decision_entries(text)
        self.assertGreater(len(entries), 0, msg="decision log must contain DEC entries")

        required_order = [
            "decision_statement",
            "chosen_option",
            "rationale_cited",
            "open_risks",
        ]
        for dec_id, block in entries:
            positions: dict[str, int] = {}
            for key in required_order:
                match = DEC_FIELD_RE[key].search(block)
                self.assertIsNotNone(match, msg=f"{dec_id} missing required field: {key}")
                positions[key] = match.start()

            observed_positions = [positions[key] for key in required_order]
            self.assertEqual(
                observed_positions,
                sorted(observed_positions),
                msg=(
                    f"{dec_id} required fields must appear in canonical order: "
                    "Decision statement -> Chosen option -> Rationale (cited) -> Open risks"
                ),
            )


if __name__ == "__main__":
    unittest.main()
