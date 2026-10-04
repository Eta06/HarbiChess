import json
import time
from concurrent.futures import Future
from contextlib import contextmanager
from typing import ClassVar

import chess
import chess.engine
import pytest

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.training import oracle_data
from harbichess.training.oracle_data import (
    generate_game,
    generation_limit,
    neural_colour,
    read_game,
    validate_row,
)


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


def test_short_balanced_neural_colours_pair_per_root_and_legacy16_is_preserved():
    for family in (0, 1, 42):
        old = [
            neural_colour(family, game, balanced=True, short_history=False) for game in range(16)
        ]
        assert old == [(family + game // 4) % 2 == 0 for game in range(16)]
        assert old[2] == old[3] and old[6] == old[7] and old[2] != old[6]
        current = [
            neural_colour(family, game, balanced=True, short_history=True) for game in range(4)
        ]
        assert current[2] != current[3]
        assert current[2] == (family % 2 == 0)
        assert all(
            neural_colour(family, game, balanced=False, short_history=True) == (family % 2 == 0)
            for game in range(4)
        )


def test_actual_native_jobs_pair_colours_and_refuse_mixed_legacy_resume(tmp_path, monkeypatch):
    class ImmediatePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, job):
            future = Future()
            try:
                future.set_result(function(job))
            except Exception as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(oracle_data, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: ExactPacketEngine())
    weights, engine = tmp_path / "model.safetensors", tmp_path / "engine"
    save_weights(weights, TorchChessNetwork())
    engine.write_text("deterministic engine fixture")
    openings = tmp_path / "book.json"
    oracle_data.publish_json(
        openings,
        {
            "seed": 7,
            "splits": {
                "train": [{"family": 0, "opening": {"moves": ["e2e4", "e7e5"]}}],
                "validation": [],
            },
        },
    )
    directory = tmp_path / "short"
    result = oracle_data.generate(
        directory,
        openings,
        weights,
        engine,
        games_per_family=4,
        workers=1,
        balanced=True,
        continuation_plies=3,
    )
    assert result["games"] == 4 and result["rows"] == 12
    jobs = [read_game(directory / f"train-000-{i}.json.gz")["job"] for i in range(4)]
    assert [job["actor"] for job in jobs] == [
        "engine-engine",
        "engine-engine",
        "neural-engine",
        "neural-engine",
    ]
    assert jobs[2]["neural_color"] != jobs[3]["neural_color"]
    assert all(job["max_plies"] == 5 for job in jobs)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["collection_plan"] == "balanced-true-history-v4"
    assert metadata["neural_colour_schedule"] == "opposite-colours-within-each-root-v1"
    with pytest.raises(ValueError, match="resume requires identical"):
        oracle_data.generate(
            directory, openings, weights, engine, games_per_family=4, workers=1, balanced=True
        )

    # The old empty-terminal-root case also keeps its original metadata/jobs.
    legacy_book = tmp_path / "legacy-book.json"
    oracle_data.publish_json(
        legacy_book,
        {
            "seed": 7,
            "splits": {
                "train": [{"family": 0, "opening": {"moves": ["f2f3", "e7e5", "g2g4", "d8h4"]}}],
                "validation": [],
            },
        },
    )
    legacy = tmp_path / "legacy"
    oracle_data.generate(
        legacy, legacy_book, weights, engine, games_per_family=16, workers=1, balanced=True
    )
    old_meta = json.loads((legacy / "metadata.json").read_text())
    assert old_meta["collection_plan"] == "balanced-every-ply-v2" and old_meta["max_plies"] == 160
    assert "continuation_plies" not in old_meta and "neural_colour_schedule" not in old_meta
    old_jobs = [read_game(legacy / f"train-000-{i}.json.gz")["job"] for i in range(16)]
    assert [job["neural_color"] for job in old_jobs] == [(i // 4) % 2 == 0 for i in range(16)]
    assert all("max_plies" not in job for job in old_jobs)
