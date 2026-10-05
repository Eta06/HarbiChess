"""Synthetic wiring/shape/immutability tests; no search or network forward."""

import copy
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import value

HERE = Path(__file__).parent
P = json.loads((HERE / "protocol-DRAFT.json").read_text())


def candidate():
    dep = P["critic18"]
    module = value.load(dep["path"], dep["sha256"])
    py, npstate = random.getstate(), np.random.get_state()
    try:
        model = module.ResidualCritic18(seed=7, contract={"updates": 64})
        model.step = 64  # Structural test fixture, not a fit.
        return model.candidate()
    finally:
        random.setstate(py)
        np.random.set_state(npstate)


def test_candidate_load_has_32_hidden_and_preserves_global_rng():
    py, npstate = random.getstate(), np.random.get_state()
    model = value.model_from_candidate(candidate(), 7, P)
    assert model.params["w1"].shape == (18, 32)
    assert model.params["w2"].shape == (32, 1)
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], npstate[1])
    assert np.random.get_state()[2:] == npstate[2:]


@pytest.mark.parametrize("mutation", ["shape", "nan", "schema"])
def test_candidate_tamper_rejected(mutation):
    c = candidate()
    if mutation == "shape":
        c["params"]["w2"].pop()
    elif mutation == "nan":
        c["params"]["w1"][0][0] = float("nan")
    else:
        c["schema"] = "old-unqualified-schema"
    with pytest.raises(ValueError):
        value.model_from_candidate(c, 7, P)


def compiled_module():
    import sys

    directory = Path(P["compiled18"]["path"]).parent
    sys.path.insert(0, str(directory))
    try:
        return value.load(P["compiled18"]["path"], P["compiled18"]["sha256"])
    finally:
        sys.path.pop(0)


def test_compiled_payload_layout_and_nonterminal_wiring_without_forward(monkeypatch):
    module = compiled_module()
    model = value.model_from_candidate(candidate(), 7, P)
    model.params["w2"][0, 0] = 0.25
    feature_calls = []
    board = chess.Board()
    for move in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push_uci(move)
    before = board.fen(), tuple(board.move_stack)

    def features(b):
        feature_calls.append(b)
        return list(range(18))

    def fake_forward(x, payload):
        assert x == list(range(18))
        assert len(payload) == 5280
        assert len(struct.unpack("=660d", payload)) == 660
        return -0.125

    monkeypatch.setattr(module, "forward", fake_forward)
    evaluator = module.CompiledEvaluator18(
        model, classical_features=features, prior_weights=[0] * 18,
        feature_scales=[1] * 18, prior_score_scale=600,
    )
    assert evaluator.nonterminal(board) == -0.125
    assert feature_calls == [board]
    assert before == (board.fen(), tuple(board.move_stack))


def test_compiled_constructor_rejects_wrong_hidden_shape():
    module = compiled_module()
    model = value.model_from_candidate(candidate(), 7, P)
    params = copy.deepcopy(model.params)
    params["b1"] = np.zeros(31)
    with pytest.raises(ValueError):
        module.CompiledEvaluator18(
            SimpleNamespace(params=params), classical_features=lambda _: [0] * 18,
            prior_weights=[0] * 18, feature_scales=[1] * 18, prior_score_scale=600,
        )


def test_search_and_controls_are_original_source_bytes():
    old = Path(P["original_search"]["path"])
    assert (HERE / "search.py").read_bytes() == old.read_bytes()
    assert value.sha(P["original_mixed_value"]["path"]) == P["original_mixed_value"]["sha256"]
    assert P["search_nodes"] == P["stockfish_nodes"] == 512
    assert P["quiescence_plies"] == 2 and P["max_depth"] == 8
    assert P["total_games"] == 160


@pytest.mark.parametrize("mutation", ["adam_shape", "adam_negative", "step", "rng"])
def test_full_native_tamper_rejected_and_restore_is_readonly(mutation):
    dep = P["critic18"]
    module = value.load(dep["path"], dep["sha256"])
    py, npstate = random.getstate(), np.random.get_state()
    try:
        model = module.ResidualCritic18(seed=7, contract={"updates": 64})
        native = model.native()
        if mutation == "adam_shape":
            native["adam_m"]["b1"].pop()
        elif mutation == "adam_negative":
            native["adam_v"]["b2"][0] = -1.0
        elif mutation == "step":
            native["step"] = 65
        else:
            native["sampler_rng"] = ["invalid"]
        with pytest.raises((ValueError, TypeError, IndexError)):
            module.ResidualCritic18(seed=7, contract={"updates": 64}, state=native)
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
