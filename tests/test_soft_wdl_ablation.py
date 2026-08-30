from types import SimpleNamespace

import mlx.core as mx

from harbichess.training.soft_wdl_ablation import (
    _mgda_weights,
    _pearson_loss,
    _soft_targets,
)


def _row(state: str, outcome: int):
    return SimpleNamespace(root_fen=state, moves=(), outcome_value=outcome)


def test_repeated_state_outcomes_become_one_shared_soft_target() -> None:
    historical = (_row("same", 0), _row("old", 1))
    fresh = (_row("same", 1), _row("new", -1))

    old_targets, fresh_targets, metrics = _soft_targets(historical, fresh)
    mx.eval(old_targets, fresh_targets)

    assert old_targets.tolist() == [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]
    assert fresh_targets.tolist() == [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    assert metrics["ambiguous_fit_states"] == 1
    assert metrics["ambiguous_fit_rows"] == 2


def test_frank_wolfe_mgda_finds_symmetric_identity_solution() -> None:
    weights = _mgda_weights([[1.0, 0.0], [0.0, 1.0]])

    assert weights == [0.5, 0.5]


def test_pearson_loss_rewards_correct_expected_score_order() -> None:
    targets = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ordered = mx.array([[4.0, 0.0, -4.0], [0.0, 4.0, 0.0], [-4.0, 0.0, 4.0]])
    reversed_logits = ordered[::-1]
    good = _pearson_loss(ordered, targets)
    bad = _pearson_loss(reversed_logits, targets)
    mx.eval(good, bad)

    assert float(good.item()) < float(bad.item())
