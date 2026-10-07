"""Cached ancestry scope regression; deterministic fake evaluator only."""

import dataclasses
import unittest

import chess
import search
import search_v2
from test_search import static


class ScopeTests(unittest.TestCase):
    def test_packets_and_ordered_evaluator_histories_exact(self):
        boards = [chess.Board(), chess.Board("8/8/8/8/8/2k5/7R/6K1 w - - 0 1")]
        b = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["e2e4", "a7a6"]:
            b.push_uci(uci)
        boards.append(b)
        for extensions in (0, 1):
            for board in boards:
                traces = []
                packets = []
                for module in (search, search_v2):
                    trace = []

                    def evaluate(b, trace=trace):
                        value = static(b)
                        trace.append((b.fen(), tuple(m.uci() for m in b.move_stack), value.hex()))
                        return value

                    engine = module.BudgetSearch(evaluate, nodes=512, incheck_extensions=extensions)
                    packets.append(dataclasses.asdict(engine.search(board)))
                    traces.append(trace)
                    if module is search_v2:
                        self.assertIsNone(engine._active_board)
                        self.assertEqual(
                            engine._key(board, 0, ("n",)), engine.key(board, 0, ("n",))
                        )
                self.assertEqual(packets[0], packets[1])
                self.assertEqual(traces[0], traces[1])

    def test_external_history_identity_and_root_reset(self):
        a = chess.Board()
        b = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            b.push_uci(uci)
        engine = search_v2.BudgetSearch(static, nodes=64)
        self.assertNotEqual(engine._key(a, 0, ("n",)), engine._key(b, 0, ("n",)))
        engine.search(a)
        result = engine.search(b)
        fresh = search_v2.BudgetSearch(static, nodes=64).search(b)
        self.assertEqual(result, fresh)
        self.assertIsNone(engine._root_context)

    def test_exception_clears_context(self):
        def fail(_):
            raise ValueError("synthetic evaluator failure")

        engine = search_v2.BudgetSearch(fail)
        with self.assertRaisesRegex(ValueError, "synthetic"):
            engine.search(chess.Board())
        self.assertIsNone(engine._active_board)


if __name__ == "__main__":
    unittest.main()
