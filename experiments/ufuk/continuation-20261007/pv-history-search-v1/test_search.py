"""Independent exhaustive legal tree oracle and fake deterministic statics."""

import math
import unittest

import chess
from search import Bound, BudgetSearch, Entry


def static(board):
    # Rule-independent deterministic bounded score, deliberately not an engine.
    n = sum(
        (p.piece_type * 7 + square) * (1 if p.color == board.turn else -1)
        for square, p in board.piece_map().items()
    )
    return n / 2048


def terminal(board, ply):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    value = 2.0 - min(ply, 10000) * 0.00001
    return value if outcome.winner == board.turn else -value


def exhaustive(board, depth, qleft, checks, ply):
    outcome = terminal(board, ply)
    if outcome is not None:
        return outcome
    if depth:
        values = []
        for move in board.legal_moves:
            board.push(move)
            try:
                values.append(-exhaustive(board, depth - 1, qleft, checks, ply + 1))
            finally:
                board.pop()
        return max(values)
    checked = board.is_check()
    if qleft == 0 and (not checked or checks == 0):
        return static(board)
    values = [] if checked else [static(board)]
    for move in board.legal_moves:
        if not checked and not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        try:
            values.append(
                -exhaustive(
                    board, 0, max(0, qleft - 1), checks - 1 if qleft == 0 else checks, ply + 1
                )
            )
        finally:
            board.pop()
    return max(values)


def roots():
    return [
        chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 0 1"),
        chess.Board("7k/6p1/6K1/8/8/8/8/R7 b - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"),  # in check
        chess.Board("7k/P7/6K1/8/8/8/8/8 w - - 0 1"),  # promotions
        chess.Board("7k/8/6K1/8/3pP3/8/8/R7 b - e3 0 1"),  # EP
    ]


class Tests(unittest.TestCase):
    def test_exact_value_against_independent_full_width_oracle(self):
        for board in roots():
            self.assertTrue(board.is_valid(), board.fen())
            for depth in (1, 2):
                scores = {}
                for move in list(board.legal_moves):
                    board.push(move)
                    try:
                        scores[move] = -exhaustive(board, depth - 1, 1, 1, 1)
                    finally:
                        board.pop()
                before = (board.fen(), tuple(board.move_stack))
                for pvs, tt in ((False, False), (True, False), (True, True)):
                    calls = []

                    def fake(b, calls=calls):
                        calls.append(b.fen())
                        return static(b)

                    result = BudgetSearch(
                        fake,
                        nodes=100000,
                        quiescence_plies=1,
                        max_depth=depth,
                        use_tt=tt,
                        use_pvs=pvs,
                    ).search(board)
                    self.assertEqual(result.completed_depth, depth)
                    self.assertFalse(result.budget_exhausted)
                    self.assertEqual(result.value.hex(), max(scores.values()).hex())
                    self.assertEqual(scores[result.move].hex(), result.value.hex())
                    self.assertEqual(result.evaluations, len(calls))
                    self.assertEqual(before, (board.fen(), tuple(board.move_stack)))

    def test_typed_bounds_no_depth_or_history_leak(self):
        search = BudgetSearch(static, nodes=1000)
        board = chess.Board()
        key = search.key(board, 0, ("n",))
        search.tt[key] = Entry(2, 0.25, Bound.LOWER, None)
        self.assertIsNone(search.probe(key, 1, -1, 1)[0])
        self.assertIsNone(search.probe(key, 2, -1, 1)[0])
        self.assertEqual(search.probe(key, 2, -1, 0.2)[0], 0.25)
        search.tt[key] = Entry(2, -0.25, Bound.UPPER, None)
        self.assertEqual(search.probe(key, 2, -0.2, 1)[0], -0.25)
        self.assertEqual(search.probe(key, 2, -1, 1)[1:], (-1, 1))
        repeated = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            repeated.push_uci(uci)
        same = chess.Board(repeated.fen())
        self.assertNotEqual(search.key(repeated, 0, ("n",)), search.key(same, 0, ("n",)))
        self.assertIsNotNone(repeated.outcome(claim_draw=True))
        self.assertIsNone(same.outcome(claim_draw=True))
        changed = chess.Board()
        changed.castling_rights = 0
        self.assertNotEqual(key, search.key(changed, 0, ("n",)))

    def test_512_budget_all_root_and_no_false_depth(self):
        board = chess.Board()
        calls = []
        result = BudgetSearch(
            lambda b: calls.append((b.fen(), tuple(b.move_stack))) or static(b), nodes=21
        ).search(board)
        self.assertEqual((result.nodes, result.root_actions, result.evaluations), (21, 20, 20))
        self.assertEqual(result.completed_depth, 0)
        self.assertEqual(result.completed_root_passes, 0)
        self.assertTrue(result.budget_exhausted)
        children = []
        for move in board.legal_moves:
            board.push(move)
            children.append(board.fen())
            board.pop()
        self.assertEqual(set(x[0] for x in calls), set(children))
        result = BudgetSearch(static, nodes=512).search(board)
        self.assertLessEqual(result.nodes, 512)
        self.assertEqual(result.root_actions, 20)
        self.assertLessEqual(result.evaluations, result.nodes)
        with self.assertRaises(ValueError):
            BudgetSearch(static, nodes=20).search(board)

    def test_check_evasion_cap_and_castling_ep_keys(self):
        checked = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        q = BudgetSearch(static, nodes=1000, quiescence_plies=0, incheck_extensions=1)
        score = q.quiesce(checked, -math.inf, math.inf, 0, 0, 1)
        expected = exhaustive(checked, 0, 0, 1, 0)
        self.assertEqual(score.hex(), expected.hex())
        self.assertEqual(q.q_check_extensions, 1)
        self.assertGreater(q.nodes, 1)
        ep = chess.Board("7k/8/6K1/8/3pP3/8/8/R7 b - e3 0 1")
        noep = ep.copy()
        noep.ep_square = None
        self.assertNotEqual(q.key(ep, 0, ("n",)), q.key(noep, 0, ("n",)))
        castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        calls = []
        result = BudgetSearch(lambda b: calls.append(b.fen()) or static(b), nodes=512).search(
            castle
        )
        self.assertEqual(result.root_actions, castle.legal_moves.count())
        for uci in ("e1g1", "e1c1"):
            child = castle.copy()
            child.push_uci(uci)
            self.assertIn(child.fen(), calls)

    def test_terminal_priorities_draw_claims_and_source_history_unchanged(self):
        histories = []
        for cycles in (1, 2, 4):
            b = chess.Board()
            for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * cycles:
                b.push_uci(uci)
            histories.append(b)
        histories += [chess.Board(f"7k/8/6K1/8/8/8/8/R7 w - - {n} 1") for n in (99, 100, 150)]
        histories += [
            chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 150 1"),
            chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("7k/8/6K1/8/8/8/8/8 b - - 0 1"),
        ]
        for board in histories:
            before = board.fen(), tuple(board.move_stack)
            self.assertEqual(BudgetSearch.terminal(board, 3), terminal(board, 3))
            if terminal(board, 0) is not None:
                result = BudgetSearch(lambda _: self.fail("terminal cannot evaluate")).search(board)
                self.assertEqual(result.nodes, 1)
                self.assertEqual(result.evaluations, 0)
                self.assertIsNone(result.move)
            self.assertEqual(before, (board.fen(), tuple(board.move_stack)))


if __name__ == "__main__":
    unittest.main()
