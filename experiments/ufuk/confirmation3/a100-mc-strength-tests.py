"""Pure synthetic analysis tests: no games, training, teacher or benchmarks."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "analysis", Path(__file__).with_name("a100-mc-strength-analysis.py")
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def book():
    return {
        "splits": {
            "arena": [{"source_game": str(i), "opening": {"moves": ["e2e4"]}} for i in range(48)]
        }
    }


def result(scores=(1, 1), cap=False):
    return {
        "games": [
            {
                "opening_pair": r,
                "candidate_color": c,
                "opening": ["e2e4"],
                "score": scores[k],
                "termination": "max_plies" if cap else "checkmate",
            }
            for r in range(48)
            for k, c in enumerate(("white", "black"))
        ]
    }


class Tests(unittest.TestCase):
    def test_pairing_by_keys_survives_order(self):
        r = result((1, 0))
        r["games"].reverse()
        s, c = a.block_scores(r, book())
        np.testing.assert_equal(s[:, 0], 1)
        np.testing.assert_equal(s[:, 1], 0)

    def test_duplicate_pair_rejected(self):
        r = result()
        r["games"][-1] = r["games"][0]
        with self.assertRaises(ValueError):
            a.block_scores(r, book())

    def test_partial_rejected(self):
        r = result()
        r["games"].pop()
        with self.assertRaises(ValueError):
            a.block_scores(r, book())

    def test_root_mismatch_rejected(self):
        r = result()
        r["games"][0]["opening"] = ["d2d4"]
        with self.assertRaises(ValueError):
            a.block_scores(r, book())

    def test_unknown_cap_must_half(self):
        with self.assertRaises(ValueError):
            a.block_scores(result((1, 1), True), book())

    def test_equal_reference_delta_exact_zero(self):
        r = a.assess_seed(result(), result((0.5, 0)), result((0.5, 0)), book(), 20261205)
        self.assertEqual(r["sf_paired_delta"]["ci_adjusted_98_75"], [0, 0])
        self.assertFalse(r["strength_pass"])

    def test_identical_colour_blocks_no_individual_resampling(self):
        r = a.assess_seed(result((1, 0)), result((1, 0)), result((0, 0)), book(), 20261205)
        self.assertEqual(r["direct"]["ci_adjusted_98_75"], [0.5, 0.5])

    def test_large_signal_pass(self):
        r = a.assess_seed(result(), result(), result((0, 0)), book(), 20261205)
        self.assertTrue(r["strength_pass"])

    def test_caps_adversarial_not_fake_draw(self):
        r = a.assess_seed(result(), result((0.5, 0.5), True), result((0, 0)), book(), 20261205)
        self.assertEqual(r["sf_delta_adversarial_caps"]["mean"], 0)
        self.assertFalse(r["strength_pass"])

    def test_repeated_analysis_bitwise(self):
        args = (result(), result((1, 0.5)), result((0, 0.5)), book(), 20261206)
        self.assertEqual(a.assess_seed(*args), a.assess_seed(*args))

    def test_exact_boundary_does_not_pass_prior_strict_direct(self):
        # Pure synthetic score mean .60 is impossible on96 WDL; threshold is still strict.
        r = a.assess_seed(result((0.5, 0.5)), result(), result((0, 0)), book(), 20261205)
        self.assertFalse(r["gates"]["direct_mean_gt_060"])


class ControllerTests(unittest.TestCase):
    def test_command_fixes_all_compute_and_history_options(self):
        spec = importlib.util.spec_from_file_location(
            "controller",
            Path(__file__).with_name("a100-mc-final-two-arm-controller.py"),
        )
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        cmd = c.command(
            Path("/p"),
            Path("/final"),
            "stockfish",
            Path("/book"),
            Path("/sf"),
            20261205,
            Path("/out"),
            3600,
        )
        for option, value in [
            ("--simulations", "16"),
            ("--nodes", "512"),
            ("--max-plies", "400"),
            ("--threads", "1"),
            ("--opening-pairs", "48"),
            ("--candidate-root-actions", "4"),
            ("--opponent-root-actions", "4"),
            ("--openings", "/book"),
        ]:
            self.assertEqual(cmd[cmd.index(option) + 1], value)
        self.assertIn("--record-engine-nodes", cmd)
        self.assertNotIn("--candidate-policy-only", cmd)

    def test_full_history_terminal_perspective_and_nodes(self):
        b = {"splits": {"arena": [{"opening": {"moves": ["f2f3", "e7e5"]}}]}}
        r = {
            "games": [
                {
                    "opening_pair": 0,
                    "candidate_color": "white",
                    "moves": ["f2f3", "e7e5", "g2g4", "d8h4"],
                    "plies": 4,
                    "termination": "checkmate",
                    "score": 0,
                    "stockfish_nodes_by_move": [{"ply": 4, "nodes": 512}],
                }
            ]
        }
        a.audit_trajectories(r, b, stockfish=True)
        r["games"][0]["score"] = 1
        with self.assertRaises(ValueError):
            a.audit_trajectories(r, b, stockfish=True)

    def test_illegal_history_rejected(self):
        b = {"splits": {"arena": [{"opening": {"moves": []}}]}}
        r = {
            "games": [
                {
                    "opening_pair": 0,
                    "candidate_color": "white",
                    "moves": ["e2e5"],
                    "plies": 1,
                    "termination": "max_plies",
                    "score": 0.5,
                }
            ]
        }
        with self.assertRaises(ValueError):
            a.audit_trajectories(r, b)

    def test_root_budget_mismatch_rejected(self):
        # Validation cannot silently substitute arena defaults16rootactions.
        r = {
            "opening_source_sha256": a.digest(__file__),
            "candidate_sha256": "x",
            "opponent_sha256": "y",
            "neural_simulations_per_move": 16,
            "neural_threads": 1,
            "candidate_root_actions": 16,
            "opponent_root_actions": 4,
        }
        with self.assertRaises(ValueError):
            a.validate_arm(r, Path(__file__), "x", "y")


if __name__ == "__main__":
    unittest.main()
