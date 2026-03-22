from __future__ import annotations

import unittest
from pathlib import Path

from gf01.formal_loader import FormalArtifactError, instance_from_formal_dict, load_formal_bundle
from gf01.models import InterventionAtom
from gf01.verifier import validate_certificate_structure


ROOT = Path(__file__).resolve().parents[1]


class TestFormalLoader(unittest.TestCase):
    def test_load_formal_bundle_from_hoa_task(self) -> None:
        instances, meta = load_formal_bundle(
            str(ROOT / "tests" / "fixtures" / "formal_toy" / "task.json")
        )
        self.assertEqual(len(instances), 1)
        instance = instances[0]
        self.assertEqual(instance.effect_ap, "out0")
        self.assertEqual(instance.automaton.input_aps, ["in0"])
        self.assertEqual(instance.provenance.get("source_type"), "tempo_hoa_trace")
        self.assertEqual(meta.get("normalization_version"), "gf01.normalization.v1")

    def test_semantic_identity_ignores_seed_and_budget_atoms(self) -> None:
        base_payload = {
            "automaton": {
                "states": ["s0"],
                "initial_state": "s0",
                "input_aps": ["in0"],
                "output_aps": ["out0"],
                "transitions": {
                    "s0": {
                        "in0=0": {"next_state": "s0", "output": {"out0": 0}},
                        "in0=1": {"next_state": "s0", "output": {"out0": 1}},
                    }
                },
            },
            "base_trace": [{"in0": 0}, {"in0": 0}],
            "effect_ap": "out0",
            "t_star": 1,
            "mode": "hard",
            "window_size": 1,
            "budget_timestep": 1,
        }
        first = instance_from_formal_dict({**base_payload, "seed": 11, "budget_atoms": 2})
        second = instance_from_formal_dict({**base_payload, "seed": 999, "budget_atoms": 99})
        self.assertEqual(first.instance_id, second.instance_id)
        self.assertEqual(first.content_hash(), second.content_hash())

    def test_rejects_non_total_transition_domain(self) -> None:
        bad_payload = {
            "automaton": {
                "states": ["s0"],
                "initial_state": "s0",
                "input_aps": ["in0"],
                "output_aps": ["out0"],
                "transitions": {
                    "s0": {
                        "in0=0": {"next_state": "s0", "output": {"out0": 0}},
                    }
                },
            },
            "base_trace": [{"in0": 0}],
            "effect_ap": "out0",
            "t_star": 0,
            "mode": "hard",
            "window_size": 1,
            "budget_timestep": 1,
        }
        with self.assertRaises(FormalArtifactError):
            instance_from_formal_dict(bad_payload)

    def test_certificate_structure_ignores_legacy_atom_budget(self) -> None:
        payload = {
            "automaton": {
                "states": ["s0"],
                "initial_state": "s0",
                "input_aps": ["in0"],
                "output_aps": ["out0"],
                "transitions": {
                    "s0": {
                        "in0=0": {"next_state": "s0", "output": {"out0": 0}},
                        "in0=1": {"next_state": "s0", "output": {"out0": 1}},
                    }
                },
            },
            "base_trace": [{"in0": 0}, {"in0": 0}],
            "effect_ap": "out0",
            "t_star": 1,
            "mode": "hard",
            "window_size": 1,
            "budget_timestep": 2,
            "budget_atoms": 1
        }
        instance = instance_from_formal_dict(payload)
        ok, reason = validate_certificate_structure(
            instance,
            [
                InterventionAtom(0, "in0", 1),
                InterventionAtom(1, "in0", 1),
            ],
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
