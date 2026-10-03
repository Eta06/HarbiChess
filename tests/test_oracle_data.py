import copy

import chess
import chess.engine
import pytest
import torch

from harbichess.chess.actions import move_to_action
from harbichess.training.oracle_data import (
    complete_multipv,
    prepare_rows,
    soft_policy,
    validate_row,
)


def row():
    board = chess.Board()
    board.push_uci("e2e4")
    move = chess.Move.from_uci("e7e5")
    return {
        "root_fen": chess.STARTING_FEN,
        "moves": ["e2e4"],
        "fen": board.fen(),
        "legal": [[m.uci(), move_to_action(board, m)] for m in board.legal_moves],
        "policy": [[move.uci(), move_to_action(board, move), 1.0]],
        "wdl": [0.2, 0.5, 0.3],
    }


def test_soft_native_wdl_does_not_turn_into_observed_integer_label():
    batch = prepare_rows([row()])
    assert batch.wdl.shape == (1, 3)
    assert batch.wdl.dtype == torch.float32
    assert batch.wdl.tolist()[0] == pytest.approx([0.2, 0.5, 0.3])
    assert batch.policies.sum() == 1
    assert not batch.policies[~batch.legal_masks].any()
    assert batch.inputs.shape == (1, 8, 8, 104)


@pytest.mark.parametrize("fault", ["illegal", "fen", "probability", "support"])
def test_corrupt_oracle_targets_and_history_are_rejected(fault):
    data = copy.deepcopy(row())
    if fault == "illegal":
        data["moves"] = ["e2e5"]
    elif fault == "fen":
        data["fen"] = chess.STARTING_FEN
    elif fault == "probability":
        data["wdl"] = [float("nan"), 0, 1]
    else:
        data["policy"][0][1] = 4671
    with pytest.raises(ValueError):
        validate_row(data)


def packet(depth, index, move, **extra):
    return {
        "depth": depth,
        "multipv": index,
        "pv": [chess.Move.from_uci(move)],
        "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
        "wdl": chess.engine.PovWdl(chess.engine.Wdl(100, 800, 100), chess.WHITE),
        **extra,
    }


def test_multipv_requires_full_unbounded_common_depth():
    packets = [packet(5, 1, "e2e4"), packet(5, 2, "d2d4"), packet(6, 1, "e2e4")]
    assert [p["depth"] for p in complete_multipv(packets, 2)] == [5, 5]
    packets.append(packet(6, 2, "d2d4", lowerbound=True))
    assert [p["depth"] for p in complete_multipv(packets, 2)] == [5, 5]
    with pytest.raises(ValueError, match="no coherent"):
        complete_multipv(packets[2:], 2)
    packets.append(packet(6, 2, "d2d4"))
    assert [p["depth"] for p in complete_multipv(packets, 2)] == [6, 6]


def test_policy_temperature_and_large_mate_scores_are_finite():
    refs = [{"cp_or_mate_score": 10000}, {"cp_or_mate_score": -10000}]
    probabilities = soft_policy(refs)
    assert sum(probabilities) == pytest.approx(1)
    assert all(0 <= p <= 1 for p in probabilities)
