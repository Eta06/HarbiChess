"""Pure conjunction regression tests; synthetic numbers are not outcomes."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "join", Path(__file__).with_name("a100-mc-all-gates-qualification.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    row = {
        "strength_pass": True,
        "gates": {"synthetic": True},
        "direct": {"mean": 0.7, "ci_adjusted_98_75": [0.6, 0.8]},
        "sf_paired_delta": {"mean": 0.2, "ci_adjusted_98_75": [0.1, 0.3]},
        "final_sf_mean": 0.3,
        "caps": {"direct": 0, "final_sf": 0, "initial_sf": 0},
        "direct_adversarial_caps": {"ci_adjusted_98_75": [0.6, 0.8]},
        "sf_delta_adversarial_caps": {"ci_adjusted_98_75": [0.1, 0.3]},
    }
    return {
        "replicated_strength_pass": True,
        "seed_results": [
            dict(copy.deepcopy(row), seed=seed) for seed in (20261205, 20261206)
        ],
    }


class Tests(unittest.TestCase):
    def test_both_complete_numeric_conjunction(self):
        m.require_strength_gates(fixture())

    def test_empty_or_missing_replication_never_passes(self):
        for rows in ([], fixture()["seed_results"][:1]):
            with self.assertRaises(AssertionError):
                m.require_strength_gates(
                    {"replicated_strength_pass": True, "seed_results": rows}
                )

    def test_true_flags_cannot_hide_failed_direct_sf_or_caps(self):
        for field, value in (
            ("direct", {"mean": 0.60, "ci_adjusted_98_75": [0.6, 0.8]}),
            ("sf_paired_delta", {"mean": 0.10, "ci_adjusted_98_75": [0.1, 0.3]}),
            ("caps", {"direct": 0.06, "final_sf": 0, "initial_sf": 0}),
        ):
            report = fixture()
            report["seed_results"][1][field] = value
            with self.assertRaises(AssertionError):
                m.require_strength_gates(report)


if __name__ == "__main__":
    unittest.main()
