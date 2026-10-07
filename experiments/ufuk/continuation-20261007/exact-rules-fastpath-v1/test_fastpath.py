"""Synthetic legal histories and fake static values only; no model/data input."""

import hashlib
import json
import random
import statistics
import time
import unittest
from pathlib import Path

import chess
from fastpath import BudgetSearch, RepetitionBoard, original

HERE = Path(__file__).resolve().parent


def history(moves, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for uci in moves.split():
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
    return board


def cases():
    boards = [chess.Board()]
    # Prospective third occurrence, current threefold, automatic fivefold.
    for n in (1, 2, 4):
        boards.append(history("g1f3 g8f6 f3g1 f6g8 " * n))
    boards += [
        history("e2e4 a7a6 e4e5 d7d5"),  # legal EP
        history("e2e4 e7e5 g1f3 b8c6 f1e2 g8f6 e1g1"),
        history("a7a8n", "7k/P7/8/8/8/8/8/K7 w - - 0 1"),
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),  # stalemate
        chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 150 1"),  # mate before75
        chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"),
        chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 99 1"),
        chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 100 1"),
        chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 150 1"),
    ]
    rng = random.Random(731)
    for _ in range(12):
        board = chess.Board()
        for ply in range(88):
            if board.outcome(claim_draw=True):
                break
            move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
            board.push(move)
            if ply in (15, 47, 87):
                boards.append(board.copy(stack=True))
    return boards


def packet(search, board):
    aliases = []

    def fake(b):
        aliases.append((b.fen(), tuple(m.uci() for m in b.move_stack)))
        # Exact deterministic arithmetic, finite mover score; no chess strength.
        v = (b.occupied.bit_count() % 7 - 3) / 16
        return v if b.turn else -v

    result = search(fake, nodes=192, max_depth=4).search(board)
    return (
        None if result.move is None else result.move.uci(),
        result.value.hex(),
        result.nodes,
        result.evaluations,
        result.completed_depth,
        result.root_actions,
        aliases,
        sorted(set(aliases)),
    )


class Tests(unittest.TestCase):
    def test_clinical_outcome_and_history(self):
        for board in cases():
            fast = RepetitionBoard.from_board(board)
            before = (fast.fen(), tuple(fast.move_stack), dict(fast._repetition_counts))
            for claims in (False, True):
                self.assertEqual(
                    board.outcome(claim_draw=claims), fast.outcome(claim_draw=claims)
                )
            self.assertEqual(
                board.can_claim_threefold_repetition(),
                fast.can_claim_threefold_repetition(),
            )
            for count in (2, 3, 5):
                self.assertEqual(board.is_repetition(count), fast.is_repetition(count))
            for move in list(board.legal_moves)[:10]:
                board.push(move)
                fast.push(move)
                self.assertEqual(board.fen(), fast.fen())
                self.assertEqual(list(board.legal_moves), list(fast.legal_moves))
                self.assertEqual(
                    board.outcome(claim_draw=True), fast.outcome(claim_draw=True)
                )
                self.assertEqual(board.is_repetition(5), fast.is_repetition(5))
                self.assertEqual(board.pop(), fast.pop())
            self.assertEqual(
                before,
                (fast.fen(), tuple(fast.move_stack), dict(fast._repetition_counts)),
            )
            copy = fast.copy(stack=True)
            self.assertEqual(
                copy.outcome(claim_draw=True), board.outcome(claim_draw=True)
            )
            self.assertIsNot(copy._repetition_counts, fast._repetition_counts)

    def test_whole_original_search_packets_and_alias_order(self):
        for board in cases():
            state = board.fen(), tuple(board.move_stack)
            self.assertEqual(
                packet(original.BudgetSearch, board), packet(BudgetSearch, board)
            )
            self.assertEqual(state, (board.fen(), tuple(board.move_stack)))

    def test_history_is_not_fen_only_and_root_guard(self):
        repeated = history("g1f3 g8f6 f3g1 f6g8 " * 2)
        fresh = chess.Board(repeated.fen())
        self.assertNotEqual(
            RepetitionBoard.from_board(repeated).outcome(claim_draw=True),
            RepetitionBoard.from_board(fresh).outcome(claim_draw=True),
        )
        with self.assertRaises(ValueError):
            RepetitionBoard.from_board(fresh).pop()
        with self.assertRaises(ValueError):
            RepetitionBoard.from_board(fresh).copy(stack=False)


def main():
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    timings = {}
    boards = cases()[13:25]
    traces = []
    for name, search in [
        ("control1", original.BudgetSearch),
        ("candidate", BudgetSearch),
        ("control2", original.BudgetSearch),
    ]:
        samples = []
        rows = []
        for board in boards:
            first = time.perf_counter()
            rows.append(packet(search, board))
            samples.append(time.perf_counter() - first)
        timings[name] = {
            "seconds": sum(samples),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
        }
        traces.append(rows)
    assert traces[0] == traces[1] == traces[2]
    body = {
        "schema": "exact-rules-fastpath-synthetic-parity-v1",
        "status": "PASS",
        "tests": result.testsRun,
        "boards": len(cases()),
        "benchmark_boards": len(boards),
        "timings": timings,
        "first_epoch": started,
        "finished_epoch": time.time(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE / "fastpath.py", HERE / "test_fastpath.py")
        },
        "original_source_sha256": hashlib.sha256(
            Path(original.__file__).read_bytes()
        ).hexdigest(),
        "models_or_training_data_used": False,
        "strength_claim": False,
    }
    with (HERE / "result.json").open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps(body, sort_keys=True))


if __name__ == "__main__":
    main()
