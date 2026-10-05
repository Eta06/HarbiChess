import unittest
from types import SimpleNamespace

from reanalyze import replay_position, run_selected


class FakeSearch:
    def search(self, board):
        self.stack = [move.uci() for move in board.move_stack]
        return SimpleNamespace(
            value=0.375,
            nodes=8192,
            evaluations=700,
            completed_depth=3,
            root_actions=20,
        )


class ReanalyzeTests(unittest.TestCase):
    def test_replays_prefix_and_preserves_mover_pov_receipt(self):
        row = {
            "row_id": "a" * 64 + ":4",
            "trajectory_id": "a" * 64,
            "root_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "prefix_uci": ["e2e4", "e7e5"],
        }
        engine = FakeSearch()
        roots = run_selected([row], lambda: engine, expected_count=1)
        self.assertEqual(engine.stack, row["prefix_uci"])
        self.assertEqual(roots[0]["root_mover"], "white")
        self.assertEqual(roots[0]["target_mover"], 0.375)
        self.assertEqual(roots[0]["nodes"], 8192)

    def test_rejects_terminal_root_and_incomplete_root_count(self):
        with self.assertRaises(ValueError):
            replay_position(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                ["f2f3", "e7e5", "g2g4", "d8h4"],
            )
        with self.assertRaises(ValueError):
            run_selected([], lambda: FakeSearch(), expected_count=1024)


if __name__ == "__main__":
    unittest.main()
