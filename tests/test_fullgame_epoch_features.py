"""Production import-path tests for epoch-owned encoded-feature compilation."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.training.torch_fullgame_ppo import _tensor_batch, compile_epoch_features


def make_row(rules, state):
    support = tuple(legal_action_indices(rules.inspect(state)))
    policy = tuple(1 / len(support) for _ in support)
    return SimpleNamespace(
        transition=SimpleNamespace(pre=state),
        legal_actions=support,
        behavior_policy=policy,
        base_policy=policy,
        action_index=0,
        advantage=-0.125,
        target_wdl=(0.0, 0.0, 1.0),
        base_value_wdl=(0.2, 0.5, 0.3),
    )


def equal_tensor_storage(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def test_same_final_FEN_distinct_full_histories_compile_separately_and_batch_exact():
    rules = PythonChessRules()
    historical = rules.initial_state()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        historical = rules.apply(historical, ChessMove(uci))
    no_history = rules.initial_state(rules.view(historical).fen)
    assert rules.view(historical).fen == rules.view(no_history).fen
    assert historical != no_history and historical.moves and not no_history.moves
    rows = tuple(make_row(rules, state) for state in (historical, no_history, historical))
    encoder = BoardEncoder(rules)
    features = compile_epoch_features(rows, encoder)
    assert len(features) == 2
    assert not np.array_equal(features[historical][0], features[no_history][0])
    fake_spec = SimpleNamespace(config=SimpleNamespace(input_channels=104))
    plain = _tensor_batch(fake_spec, encoder, rules, rows, "cpu")
    cached = _tensor_batch(fake_spec, encoder, rules, rows, "cpu", features)
    for expected, actual in zip(plain, cached, strict=True):
        equal_tensor_storage(expected, actual)


def test_new_epoch_owns_new_readonly_storage_no_cross_epoch_cache_or_rng_change():
    rules = PythonChessRules()
    state = rules.initial_state()
    rows = (make_row(rules, state),) * 129
    encoder = BoardEncoder(rules)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    guarded = []
    first = compile_epoch_features(rows, encoder, guard=lambda: guarded.append(1))
    second = compile_epoch_features(rows, encoder)
    assert len(first) == 1 and len(guarded) == 2
    assert first is not second and first[state][0] is not second[state][0]
    assert not np.shares_memory(first[state][0], second[state][0])
    assert first[state][0].dtype == np.float32 and not first[state][0].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first[state][0][0] = 1.0
    expected = np.array(encoder.encode(state).values, dtype=np.float32)
    assert first[state][0].tobytes() == second[state][0].tobytes() == expected.tobytes()
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[1].tobytes() == before[1][1].tobytes()
    assert after[2:] == before[1][2:]
    equal_tensor_storage(before[2], torch.get_rng_state())


def test_compile_rejects_conflicting_support_and_honors_resource_abort():
    rules = PythonChessRules()
    state = rules.initial_state()
    row = make_row(rules, state)
    conflict = SimpleNamespace(transition=row.transition, legal_actions=row.legal_actions[::-1])
    with pytest.raises(ValueError, match="conflicting legal support"):
        compile_epoch_features((row, conflict), BoardEncoder(rules))

    def exhausted():
        raise TimeoutError("whole-run-resource-budget")

    with pytest.raises(TimeoutError, match="whole-run-resource-budget"):
        compile_epoch_features((row,), BoardEncoder(rules), guard=exhausted)
