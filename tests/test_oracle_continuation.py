import time
from contextlib import contextmanager
from typing import ClassVar

import chess
import chess.engine
import pytest

from harbichess.training.oracle_data import generate_game, generation_limit, read_game, validate_row


class ExactPacketEngine:
    id: ClassVar = {"name": "deterministic test reference, not Stockfish benchmark"}

    def configure(self, options):
        assert options["Threads"] == 1 and options["UCI_ShowWDL"]

    @contextmanager
    def analysis(self, board, limit, *, multipv, game):
        moves = sorted(board.legal_moves, key=lambda m: m.uci())[:multipv]
        packets = [
            {
                "multipv": i,
                "depth": 5,
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(20 - i), board.turn),
                "wdl": chess.engine.PovWdl(chess.engine.Wdl(200, 600, 200), board.turn),
                "nodes": limit.nodes,
            }
            for i, move in enumerate(moves, 1)
        ]

        class Analysis:
            info: ClassVar = {"nodes": limit.nodes}

            def __iter__(self):
                return iter(packets)

            def wait(self):
                pass

        yield Analysis()

    def close(self):
        pass


def test_short_native_collection_keeps_true_root_history_targets_and_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: ExactPacketEngine())
    opening = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
    job = {
        "opening": opening,
        "max_plies": len(opening) + 3,
        "seed": 7,
        "actor": "engine-engine",
        "stockfish": "fixture",
        "output": str(tmp_path / "game.json.gz"),
        "deadline": time.time() + 10,
        "family": 9,
        "game": 0,
        "split": "train",
        "label_every_ply": True,
    }
    receipt = generate_game(job)
    assert receipt["rows"] == receipt["queries"] == 3
    game = read_game(tmp_path / "game.json.gz")
    assert len(game["final_moves"]) == len(opening) + 3
    assert game["observed_result"] is None and game["termination"] == "max_plies"
    assert game["actual_nodes"] == 3 * 32768
    assert "deadline" not in game["job"] and game["job"]["max_plies"] == 13
    for i, row in enumerate(game["rows"]):
        board = validate_row(row)
        assert len(board.move_stack) == len(opening) + i
        assert row["moves"][: len(opening)] == opening
        assert row["wdl"] == [0.2, 0.6, 0.2]


def test_versioned_bad_continuation_refuses_and_legacy160_is_unchanged():
    board = chess.Board()
    assert generation_limit({}, board) == 160
    for bound in (False, 0, 513, 1.5):
        with pytest.raises(ValueError, match="bound"):
            generation_limit({"max_plies": bound}, board)
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    assert generation_limit({}, board) == 160
    with pytest.raises(ValueError, match="terminal"):
        generation_limit({"max_plies": 36}, board)
