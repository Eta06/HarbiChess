"""Three synthetic legal/gradient/native tests; no SGD or actual search."""

import copy
import json

import chess
import pytest
from learner_v2 import MATH, Learner, row_gradient
from transform import digest, old, transform


def fixture():
    groups = [f"g{i}" for i in range(16)]
    leaves = []
    history = {"root_fen": chess.STARTING_FEN, "prefix_uci": []}
    board = chess.Board()
    for g in groups:
        for i in range(16):
            target = int(i < 4)
            leaves.append(
                dict(
                    trajectory_id=g,
                    history=history,
                    history_sha256=digest(history),
                    features=old.features(board, 0.0),
                    static=0.0,
                    status="completed",
                    target=target,
                    reference_mover_value=0.2 if target else 0.0,
                )
            )
    return {"leaves": leaves}, {"train": groups, "validation": ["heldout"]}


def test_balanced_gradient_normalization_gameequal_prevalence():
    data, split = fixture()
    groups, record = transform(data, split)
    assert record["positives"] == 64 and record["game_equal_positive_prevalence"] == 0.25
    assert record["pos_weight"] == 3 and record["gradient_normalization"] == 1.5
    gradients = [row_gradient([0.0] * 16, r, 0.25)[0] for r in groups["g0"]]
    assert abs(sum(gradients)) < 1e-12
    checked = copy.deepcopy(groups["g0"][0])
    checked["features"][1] = 1
    with pytest.raises(ValueError):
        row_gradient([0.0] * 16, checked, 0.25)


def test_nonchecked_fullleaf_and_validation_censor_tampering():
    data, split = fixture()
    h = {"root_fen": "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1", "prefix_uci": []}
    board = chess.Board(h["root_fen"])
    assert board.is_check()
    checked = copy.deepcopy(data["leaves"][0])
    checked.update(
        history=h, history_sha256=digest(h), features=old.features(board, 0.0), checked=True
    )
    data["leaves"].append(checked)
    validation = copy.deepcopy(data["leaves"][0])
    validation["trajectory_id"] = "heldout"
    data["leaves"].append(validation)
    _, record = transform(data, split)
    assert record["rows"] == 256
    checked["checked"] = False
    with pytest.raises(ValueError):
        transform(data, split)
    checked["checked"] = True
    data["leaves"][0]["history"] = {"root_fen": chess.STARTING_FEN, "prefix_uci": ["e2e5"]}
    with pytest.raises(ValueError):
        transform(data, split)


def test_fresh_v2_native_allfields_roundtrip_rejects_old_schema():
    data, split = fixture()
    _, record = transform(data, split)
    contract = dict(
        schema="own-balanced-risk-learning-contract-v2",
        seed=5,
        updates=48,
        math=MATH,
        transform=record,
        inputs={},
        source_commit="0" * 40,
        helper_sha256={},
        first_epoch=100,
        deadline_epoch=1000,
    )
    fresh = Learner(5, contract).native()
    decoded = json.loads(json.dumps(fresh))
    restored = Learner(5, contract, decoded).native()
    assert json.dumps(fresh, sort_keys=True) == json.dumps(restored, sort_keys=True)
    assert fresh["weights"] == [0.0] * 16 and fresh["step"] == 0
    decoded["schema"] = "own-selective-quiescence-effort-native-v1"
    with pytest.raises(ValueError):
        Learner(5, contract, decoded)
