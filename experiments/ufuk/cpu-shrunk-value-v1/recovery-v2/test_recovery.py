"""Tiny fake evaluator/owner fixtures only; never actual arena or model qualification."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
from recovery_support import clock, match_prefix, parse_progress, sha, validate_recovery

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value))


def load(name):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(path):
    board = chess.Board()
    opening = ["e2e4", "e7e5", "g1f3", "b8c6"]
    for uci in opening:
        board.push_uci(uci)
    moves = []
    for _ in range(2):
        move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        moves.append(move.uci())
        board.push(move)
    game = {
        "opening_pair": 0,
        "candidate_color": "white",
        "opening": opening,
        "moves": opening + moves,
        "plies": 6,
        "termination": "max_plies",
        "score": 0.5,
        "search_by_move": [],
        "stockfish_nodes_by_move": [],
        "move_wall_seconds": [0.001, 0.002],
        "wall_seconds": 0.003,
    }
    events = [{"type": "game_start", "pair": 0, "color": "white"}]
    events += [{"type": "move", "ply": 5 + i, "uci": m} for i, m in enumerate(moves)]
    events += [
        {"type": "game_end", "game": game},
        {"type": "game_start", "pair": 0, "color": "black"},
        {"type": "move", "ply": 5, "uci": moves[0]},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return [{"opening": {"moves": opening}}], game, events


def test_exact_closed_game_and_partial_history_preserved(tmp_path):
    path = tmp_path / "old.jsonl"
    book, game, _ = progress(path)
    completed, raw, partial = parse_progress(path, sha(path), book, "e8")
    assert completed == {(0, "white"): game}
    assert json.loads(raw[0, "white"])["game"] == game
    assert partial == {"key": (0, "black"), "moves": [game["moves"][4]]}
    with pytest.raises(ValueError, match="SF partial"):
        parse_progress(path, sha(path), book, "SF512")


def test_clock_keeps_original_first_and_rejects_restart():
    invocation = {"original_first_epoch": 100.0, "original_deadline_epoch": 2800.0}
    assert clock(invocation, 5000.0, 200.0) == (100.0, 2800.0)
    with pytest.raises(ValueError):
        clock(invocation, 5000.0, 2800.0)
    with pytest.raises(ValueError):
        clock({**invocation, "original_deadline_epoch": 3000.0}, 5000.0, 200.0)
    assert clock({**invocation, "original_deadline_epoch": 1000.0}, 1000.0, 200.0) == (
        100.0,
        1000.0,
    )


def test_packet_mismatch_failclosed():
    match_prefix({"selected_move": "e2e4"}, "e2e4")
    with pytest.raises(ValueError, match="reproduce"):
        match_prefix({"selected_move": "e2e3"}, "e2e4")


def test_real_resume_main_wiring_fake_search_only(tmp_path, monkeypatch):
    module = load("resume_tournament")
    path = tmp_path / "old.jsonl"
    book, original_game, _ = progress(path)
    book_path = tmp_path / "book.json"
    dump(book_path, {"splits": {"arena": book}})
    candidate, opponent, engine = [tmp_path / name for name in ("model", "other", "engine")]
    for file in (candidate, opponent, engine):
        file.write_bytes(b"SYNTHETIC-FIXTURE-NOT-NN")
    protocol = tmp_path / "protocol.json"
    p = {
        "source_commit": "0" * 40,
        "match_seeds": [1],
        "book_sha256": sha(book_path),
        "stockfish_sha256": sha(engine),
        "memory_max_bytes": 1000000,
        "models": {"1": {"fixture": {"sha256": sha(candidate)}}},
        "search_nodes": 512,
        "quiescence_plies": 2,
        "max_depth": 8,
        "opening_pairs": 1,
        "max_plies": 6,
        "stockfish_nodes": 512,
    }
    dump(protocol, p)
    first = time.time() - 2
    invocation = tmp_path / "invocation.json"
    command = [
        "python",
        "OLD_tournament.py",
        "--candidate",
        str(candidate),
        "--opponent",
        str(opponent),
        "--protocol",
        str(protocol),
        "--book",
        str(book_path),
        "--stockfish",
        str(engine),
    ]
    dump(
        invocation,
        {
            "original_first_epoch": first,
            "original_deadline_epoch": first + 2700,
            "seed": 1,
            "opponent": "e8",
            "command": command,
        },
    )

    class FakeSearch:
        def __init__(self, evaluator, **kwargs):
            pass

        def search(self, board):
            return SimpleNamespace(
                move=sorted(board.legal_moves, key=lambda m: m.uci())[0],
                nodes=board.legal_moves.count() + 1,
                evaluations=1,
                completed_depth=1,
                root_actions=board.legal_moves.count(),
                value=0.0,
            )

    monkeypatch.setattr(module, "BudgetSearch", FakeSearch)
    monkeypatch.setattr(module, "NeuralValue", lambda path: None)
    monkeypatch.setattr(module, "clean_source", lambda commit: None)
    monkeypatch.setattr(
        module, "CgroupMemoryBudget", lambda limit: SimpleNamespace(check=lambda: None)
    )
    output = tmp_path / "new.json"
    args = [
        "resume_tournament.py",
        "--protocol",
        str(protocol),
        "--candidate",
        str(candidate),
        "--opponent",
        str(opponent),
        "--seed",
        "1",
        "--book",
        str(book_path),
        "--stockfish",
        str(engine),
        "--output",
        str(output),
        "--progress",
        str(tmp_path / "new.jsonl"),
        "--deadline-epoch",
        str(first + 2700),
        "--parent-progress",
        str(path),
        "--parent-progress-sha256",
        sha(path),
        "--parent-invocation",
        str(invocation),
        "--parent-invocation-sha256",
        sha(invocation),
        "--whole-deadline-epoch",
        str(first + 7200),
        "--old-parent-executor-sha256",
        "a" * 64,
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    result = json.loads(output.read_text())
    assert result["games"][0] == original_game
    assert result["games"][1]["moves"] == original_game["moves"]
    assert result["started_epoch"] == first
    assert result["recovery"]["duplicate_NN_moves_researched"] == 1
    assert (
        result["games"][1]["search_by_move"][0]["measurement_origin"]
        == "actual-recovery-prefix-research"
    )
    owner = {"original_deadline_epoch": first + 2700}
    contract = {
        "recovery_executor_sha256": sha(HERE / "resume_tournament.py"),
        "inputs": {"tournament.py": "a" * 64},
    }
    validate_recovery(result, owner, contract)
    result["helper_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="actual executor"):
        validate_recovery(result, owner, contract)
