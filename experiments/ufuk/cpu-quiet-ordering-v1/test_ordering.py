"""Only tiny synthetic learner/search fixtures; no NN or production data fit."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import chess
import pytest
from model import DIM, features
from ordered_search import search_type
from ordering_learner import Learner, row_gradient

BASE = Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py")


def base():
    spec = importlib.util.spec_from_file_location("quiet_order_test_base", BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BudgetSearch


def test_zero_weights_exact_original_order_and_all_result_fields():
    cls = base()
    new = search_type(cls)
    board = chess.Board()
    assert new(lambda _: 0).ordered(board, board.legal_moves) == cls.ordered(
        board, board.legal_moves
    )
    # Tiny fake-value32-node fixture, not a real evaluator/search qualification.
    assert new(lambda _: 0, nodes=32).search(board) == cls(lambda _: 0, nodes=32).search(board)


def test_capture_slots_unchanged_while_quiet_sort_changes():
    cls = base()
    board = chess.Board("4k3/8/8/8/3p4/8/3R4/4K3 w - - 0 1")
    weights = [0.0] * DIM
    wanted = chess.Move.from_uci("d2d3")
    for i in features(board, wanted):
        weights[i] = 1.0
    old = cls.ordered(board, board.legal_moves)
    new = search_type(cls)(lambda _: 0, ordering_weights=weights).ordered(board, board.legal_moves)
    assert {m for m in old} == {m for m in new}
    for index, move in enumerate(old):
        if board.is_capture(move) or move.promotion:
            assert new[index] == move
    assert old != new


def test_mover_relative_feature_reflection():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    mirrored = board.mirror()
    other = chess.Move.from_uci("e7e5")
    assert features(board, move) == features(mirrored, other)


def test_sparse_ce_gradient_sign_and_shared_feature_cancellation():
    row = dict(features=[(0, 7), (1, 7)], target=0)
    grad = row_gradient([0.0] * DIM, row)
    assert grad[0] == -0.5 and grad[1] == 0.5 and grad[7] == 0
    assert all(math.isfinite(x) for x in grad)


def test_synthetic_whole8_pause4_fresh8_json_and_all_rng_exact():
    groups = {"fixture": [dict(features=[(0, 7), (1, 7)], target=0)]}
    contract = dict(updates=8, source="synthetic-not-production", data="fixture")
    whole = Learner(20262905, contract)
    whole.advance(groups, 8)
    split = Learner(20262905, contract)
    split.advance(groups, 4)
    state = json.loads(json.dumps(split.native()))
    resumed = Learner(20262905, contract, state)
    resumed.advance(groups, 8)
    assert json.dumps(resumed.native(), sort_keys=True) == json.dumps(
        whole.native(), sort_keys=True
    )
    bad = copy.deepcopy(state)
    bad["contract"]["data"] = "different"
    with pytest.raises(ValueError):
        Learner(20262905, contract, bad)
    bad = copy.deepcopy(state)
    bad["v"][0] = -1
    with pytest.raises(ValueError):
        Learner(20262905, contract, bad)
