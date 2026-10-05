"""Focused negative checks for the frozen arena protocol/qualification contract."""

import copy
import json
import unittest
from pathlib import Path

from contracts import (
    load_fit_manifest,
    verify_fixed_arena_protocol,
    verify_profile_receipt,
)

HERE = Path(__file__).parent


def make_receipt(protocol, manifest):
    first = 1791200000.0
    protocol["profile_first_epoch"] = first
    protocol["profile_deadline_epoch"] = first + 600
    rows = []
    for seed in protocol["match_seeds"]:
        for role in protocol["roles"]:
            for opening in range(4):
                rows.append(
                    {
                        "seed": seed,
                        "role": role,
                        "opening": opening,
                        "model_sha256": manifest["fits"][str(seed)][role]["sha256"],
                    }
                )
    return {
        "status": "PASS-qualification-not-strength",
        "protocol_sha256": "protocol-hash",
        "fit_provenance_sha256": "fit-hash",
        "helper_sha256": "helper-hash",
        "original_first_epoch": first,
        "original_deadline_epoch": first + 600,
        "finished_epoch": first + 500,
        "rows": rows,
    }


class ArenaContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / "protocol-TEMPLATE.json").read_text())
        _, self.manifest = load_fit_manifest(self.protocol)
        self.receipt = make_receipt(self.protocol, self.manifest)

    def check_profile(self, receipt=None):
        return verify_profile_receipt(
            self.protocol,
            self.receipt if receipt is None else receipt,
            "protocol-hash",
            "fit-hash",
            "helper-hash",
            self.manifest,
        )

    def test_fixed_protocol_passes(self):
        self.assertTrue(verify_fixed_arena_protocol(self.protocol))

    def test_changed_gate_rejected(self):
        p = copy.deepcopy(self.protocol)
        p["screen"]["paired_sf_gain_over_e8_strictly_above"] = 0
        with self.assertRaises(ValueError):
            verify_fixed_arena_protocol(p)

    def test_changed_source_rejected(self):
        p = copy.deepcopy(self.protocol)
        p["source_commit"] = "0" * 40
        with self.assertRaises(ValueError):
            verify_fixed_arena_protocol(p)

    def test_registered_screen_scope_is_unambiguous(self):
        self.assertTrue(
            self.protocol["screen"]["evaluate_both_seeds_and_all_three_arms"]
        )
        self.assertNotIn("both_seeds_and_all_three_arms", self.protocol["screen"])

    def test_valid_exact_profile_inventory_passes(self):
        self.assertTrue(self.check_profile())

    def test_counterfeit_profile_helper_rejected(self):
        q = copy.deepcopy(self.receipt)
        q["helper_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check_profile(q)

    def test_wrong_profile_fit_binding_rejected(self):
        q = copy.deepcopy(self.receipt)
        q["fit_provenance_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check_profile(q)

    def test_late_profile_rejected(self):
        q = copy.deepcopy(self.receipt)
        q["finished_epoch"] = q["original_deadline_epoch"] + 0.001
        with self.assertRaises(ValueError):
            self.check_profile(q)

    def test_corrupt_profile_model_binding_rejected(self):
        q = copy.deepcopy(self.receipt)
        q["rows"][0]["model_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check_profile(q)

    def test_missing_profile_root_rejected(self):
        q = copy.deepcopy(self.receipt)
        q["rows"].pop()
        with self.assertRaises(ValueError):
            self.check_profile(q)


if __name__ == "__main__":
    unittest.main()
