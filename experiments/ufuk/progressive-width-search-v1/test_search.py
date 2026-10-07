import importlib.util
import sys
import unittest
from pathlib import Path

import chess
import search

spec = importlib.util.spec_from_file_location(
    "frozen_pvs_control",
    Path(__file__).resolve().parent.parent / "pv-history-search-v1/search_v2.py",
)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def deterministic(board):
    score = sum((i + 1) * ord(c) for i, c in enumerate(board.fen())) % 2001
    return (score - 1000) / 1000


def traced(log):
    def evaluate(position):
        log.append(
            (
                position.root().fen(),
                tuple(m.uci() for m in position.move_stack),
                position.fen(),
                deterministic(position).hex(),
            )
        )
        return deterministic(position)

    return evaluate


def plain(board, depth, ply=0):
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        magnitude = 2 - min(ply, 10000) * 0.00001
        return magnitude if outcome.winner == board.turn else -magnitude
    if depth == 0:
        return deterministic(board)
    values = []
    for move in board.legal_moves:
        board.push(move)
        try:
            values.append(-plain(board, depth - 1, ply + 1))
        finally:
            board.pop()
    return max(values)


def clinical():
    boards = [
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1"),
        chess.Board("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1"),
        chess.Board("7k/8/8/8/8/8/r7/K7 w - - 0 1"),
        chess.Board("7k/8/8/8/8/8/R7/K7 w - - 99 50"),
    ]
    repeat = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeat.push_uci(uci)
    boards.extend(
        [
            repeat,
            chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1"),
        ]
    )
    return boards


class SearchTests(unittest.TestCase):
    def test_full_width_matches_frozen_packet_and_every_fake_eval_alias(self):
        for board in clinical():
            for budget in (64, 512):
                traces = [[], []]

                def fn(which, logs=traces):
                    return traced(logs[which])

                a = old.BudgetSearch(fn(0), nodes=budget, max_depth=4).search(board)
                b = search.BudgetSearch(
                    fn(1), nodes=budget, max_depth=4, root_width="full"
                ).search(board)
                for key, value in vars(a).items():
                    observed = getattr(b, key)
                    self.assertEqual(
                        value.hex() if isinstance(value, float) else value,
                        observed.hex() if isinstance(observed, float) else observed,
                        (board.fen(), budget, key),
                    )
                self.assertEqual(traces[0], traces[1])
                self.assertLessEqual(b.nodes, budget)
                self.assertEqual(b.full_legal_completed_depth, b.completed_depth)

    def test_progressive_coverage_and_charged_budget_with_no_input_mutation(self):
        board = chess.Board()
        original = (board.fen(), list(board.move_stack))
        result = search.BudgetSearch(
            deterministic,
            nodes=20000,
            max_depth=4,
            quiescence_plies=0,
            incheck_extensions=0,
        ).search(board)
        self.assertEqual(result.root_actions, 20)
        self.assertEqual(
            [len(actions) for _, actions in result.completed_root_coverage],
            [20, 8, 4, 2],
        )
        self.assertEqual(result.full_legal_completed_depth, 1)
        self.assertEqual(result.completed_depth, 4)
        for (d, a), (e, b) in zip(
            result.completed_root_coverage,
            result.completed_root_coverage[1:],
            strict=False,
        ):
            self.assertEqual(e, d + 1)
            self.assertTrue(set(b) <= set(a))
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual((board.fen(), list(board.move_stack)), original)
        self.assertLessEqual(result.nodes, 20000)
        self.assertIn("not full-legal minimax", result.depth_semantics)

    def test_generous_full_width_matches_independent_exhaustive_depth2(self):
        for board in clinical():
            result = search.BudgetSearch(
                deterministic,
                nodes=100000,
                max_depth=2,
                quiescence_plies=0,
                incheck_extensions=0,
                root_width="full",
            ).search(board)
            self.assertEqual(result.value.hex(), plain(board, 2).hex(), board.fen())

    def test_beam_can_prune_true_deeper_best_no_minimax_claim(self):
        def trap(board):
            return (
                1.0
                if len(board.move_stack) == 2 and board.move_stack[0].uci() == "h2h4"
                else 0.0
            )

        board = chess.Board()
        settings = dict(
            nodes=100000, max_depth=2, quiescence_plies=0, incheck_extensions=0
        )
        full = search.BudgetSearch(trap, root_width="full", **settings).search(board)
        beam = search.BudgetSearch(trap, **settings).search(board)
        self.assertEqual(full.move.uci(), "h2h4")
        self.assertEqual(full.value, 1.0)
        self.assertEqual(beam.value, 0.0)
        self.assertNotIn("h2h4", beam.completed_root_coverage[-1][1])
        self.assertEqual(beam.full_legal_completed_depth, 1)
        self.assertEqual(beam.completed_depth, 2)

    def test_static_fallback_all_actions_and_incomplete_pass_disclosed(self):
        board = chess.Board()
        result = search.BudgetSearch(deterministic, nodes=21, max_depth=8).search(board)
        self.assertEqual(result.nodes, 21)
        self.assertEqual(result.evaluations, 20)
        self.assertEqual(result.root_actions, 20)
        self.assertEqual(result.completed_depth, 0)
        self.assertEqual(result.completed_root_coverage, ())
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.completed_attempted_root_actions, 0)
        self.assertEqual(len(result.attempted_root_actions), 20)

    def test_full_history_keys_and_root_legality_budget_rejected(self):
        a = chess.Board()
        b = a.copy()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            b.push_uci(uci)
        self.assertNotEqual(
            search.BudgetSearch.key(a, 0, ("n",)), search.BudgetSearch.key(b, 0, ("n",))
        )
        with self.assertRaises(ValueError):
            search.BudgetSearch(deterministic, nodes=20).search(a)
        with self.assertRaises(ValueError):
            search.BudgetSearch(deterministic, root_width="adaptive")


if __name__ == "__main__":
    unittest.main()
