import json

import chess
import chess.engine
import pytest

from harbichess.evaluation.teacher_probe import freeze_panel, last_complete_info, reference_info


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_native_reference_is_root_stm_including_mate(turn):
    board = chess.Board()
    board.turn = turn
    move = next(iter(board.legal_moves))
    info = {
        "score": chess.engine.PovScore(chess.engine.Mate(2), turn),
        "wdl": chess.engine.PovWdl(chess.engine.Wdl(1000, 0, 0), turn),
        "pv": [move],
    }
    result = reference_info(info, board)
    assert result["wdl"] == [1, 0, 0]
    assert result["expected_score"] == 1
    assert result["mate"] == 2
    board.turn = not turn
    result = reference_info(info, board)
    assert result["wdl"] == [0, 0, 1]
    assert result["expected_score"] == 0
    assert result["mate"] == -2


def test_panel_preserves_history_and_rejects_illegal_source(tmp_path):
    moves = [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
        "b5a4",
        "g8f6",
        "e1g1",
        "f8e7",
        "f1e1",
        "b7b5",
    ]
    game = {"moves": moves, "opening_pair": 0, "candidate_color": "white", "score": 0.5}
    source = tmp_path / "games.json"
    source.write_text(json.dumps({"games": [game, game]}))
    panel = freeze_panel(source)
    assert len({row["id"] for row in panel}) == len(panel)
    assert panel == freeze_panel(source)
    for row in panel:
        board = chess.Board(row["root_fen"])
        for uci in row["moves"]:
            board.push_uci(uci)
        assert board.fen() == row["fen"]
        assert board.has_castling_rights(chess.WHITE) == (row["ply"] < 9)
    game["moves"] = ["e2e5"]
    source.write_text(json.dumps({"games": [game]}))
    with pytest.raises(ValueError, match="illegal source"):
        freeze_panel(source)


def test_completed_uci_packet_ignores_stale_bounds_and_partial_aspiration():
    score = chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE)
    wdl = chess.engine.PovWdl(chess.engine.Wdl(100, 800, 100), chess.WHITE)
    pv = [chess.Move.from_uci("e2e4")]
    exact = {"score": score, "wdl": wdl, "pv": pv, "depth": 8}
    bounded = {
        **exact,
        "score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
        "lowerbound": True,
        "depth": 9,
    }
    assert last_complete_info([bounded, exact, {"nodes": 1000}, bounded]) == exact
    merged = dict(bounded)
    merged.update(exact)
    assert merged["lowerbound"]  # python-chess summary flags survive a later complete score.
    assert "lowerbound" not in last_complete_info([bounded, exact])
    with pytest.raises(ValueError, match="no completed"):
        last_complete_info([bounded])
