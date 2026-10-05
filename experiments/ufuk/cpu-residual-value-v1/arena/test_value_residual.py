"""Actual frozen checkpoints, at most three distinct boards; no search/full games."""

import importlib.util
import os
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
sys.path.insert(0, str(SOURCE / "src"))
spec = importlib.util.spec_from_file_location("residual_value_adapter", HERE / "value.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

legal_action_indices = importlib.import_module("harbichess.chess.actions").legal_action_indices
_softmax = importlib.import_module("harbichess.search.evaluator")._softmax

E8 = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/harbichess-inputs/initial-e8.safetensors"
)
FITS = Path(
    "/workspace/work/harbichess/cpu-residual-value-v1-actual/fits/20262705-residual/checkpoints"
)
ZERO = FITS / "step-00000000/model.safetensors"
FINAL = FITS / "step-00001024/model.safetensors"
LEGACY_LINEAR = Path(
    "/workspace/work/harbichess/cpu-outcome-value-v1-actual/fits/20262305-linear/checkpoints/step-00000000/model.safetensors"
)
LEGACY_SPARSE = Path(
    "/workspace/work/harbichess/cpu-shrunk-value-v1-actual/fits/20262605-shrunk/checkpoints/step-00000000/model.safetensors"
)


@pytest.fixture(scope="module")
def evaluators():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    assert torch.__version__ == "2.14.1+cpu" and not torch.cuda.is_available()
    return [adapter.NeuralValue(path) for path in (E8, ZERO, FINAL, LEGACY_LINEAR, LEGACY_SPARSE)]


def boards():
    for moves in ([], ["e2e4"], ["g1f3", "g8f6", "f3g1", "f6g8"]):
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        yield board


def inputs(evaluator, board):
    x = torch.tensor(np.asarray(evaluator.encoder.encode_board(board).values).reshape(1, 8, 8, 104))
    return x, torch.tensor([legal_action_indices(board)])


def test_zero_identity_and_nonzero_actual_full_composition_policy_and_sign(evaluators):
    e8, zero, final, _, _ = evaluators
    assert zero.value_mode == final.value_mode == "schema2-core-full-base-plus-residual-once"
    assert torch.count_nonzero(zero.model.value_sparse_head.feature.weight) == 0
    nonzero = False
    with torch.inference_mode():
        for board in boards():
            x, actions = inputs(e8, board)
            ep, ev = e8.model.masked_policy_value(x, actions)
            zp, zv = zero.model.masked_policy_value(x, actions)
            fp, fv = final.model.masked_policy_value(x, actions)
            assert torch.equal(ep, zp) and torch.equal(ep, fp)
            assert torch.equal(ev, zv)
            assert torch.equal(zero.logits(board), zv)
            assert torch.equal(final.logits(board), fv)
            delta = final.model.value_sparse_head(x)
            assert torch.equal(ev + delta, fv), "core must add residual exactly once"
            nonzero |= bool(torch.count_nonzero(delta))
            expected = _softmax(tuple(fv[0].tolist()))
            actual = _softmax(tuple(final.logits(board)[0].tolist()))
            assert actual == expected
            assert final(board) == expected[0] - expected[2]
            assert -1 <= final(board) <= 1
            assert torch.equal(ev, e8.logits(board))
    assert nonzero, "actual trained1024 residual must differ from zero control"


def test_legacy_zero_linear_and_v1_sparse_fastpaths_unchanged(evaluators):
    _, _, _, linear, sparse = evaluators
    assert linear.linear_only and linear.value_mode == "legacy-full-or-zero-linear"
    assert sparse.value_mode == "legacy-v1-sparse-only"
    board = chess.Board()
    with torch.inference_mode():
        for evaluator in (linear, sparse):
            x, actions = inputs(evaluator, board)
            full = evaluator.model.masked_policy_value(x, actions)[1]
            assert torch.equal(full, evaluator.logits(board))
            assert evaluator(board) == 0.0


def test_unknown_composition_rejected_without_inference(monkeypatch):
    class Invalid:
        def __init__(self):
            self.specification = {"value_sparse": {"schema": 2, "composition": "UNREGISTERED"}}

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

    monkeypatch.setattr(adapter, "load_weights", lambda path: Invalid())
    with pytest.raises(ValueError, match="unsupported"):
        adapter.NeuralValue(Path("unused"))
