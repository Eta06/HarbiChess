from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.continuous_policy_iteration import (  # noqa: E402
    ContinuousPolicyIterationConfig,
    _combine_policy,
    _continuous_wdl_gate,
    _policy_gate,
)
from harbichess.training.full_gumbel_transfer import PreparedTransfer  # noqa: E402


def _config(**overrides) -> ContinuousPolicyIterationConfig:
    values = {
        "output_dir": Path("output"),
        "value_result": Path("value.json"),
        "model_path": Path("model.safetensors"),
    }
    values.update(overrides)
    return ContinuousPolicyIterationConfig(**values)


def _wdl(**overrides) -> dict[str, float]:
    values = {
        "cross_entropy": 0.90,
        "macro_cross_entropy": 0.92,
        "expected_score_pearson": 0.45,
        "loss_draw_margin": 0.20,
        "win_draw_margin": 0.20,
        "ece_10": 0.03,
    }
    values.update(overrides)
    return values


def test_config_rejects_impossible_rolling_window() -> None:
    with pytest.raises(ValueError, match="rolling generations"):
        _config(updates=2, rolling_generations=3)


def test_policy_gate_requires_imitation_gain_without_top_action_regression() -> None:
    before = {"cross_entropy": 2.0, "top_action_agreement": 0.25}

    assert _policy_gate(before, {"cross_entropy": 1.98, "top_action_agreement": 0.30}) == ()
    assert len(_policy_gate(before, {"cross_entropy": 1.995, "top_action_agreement": 0.20})) == 2


def test_continuous_wdl_gate_keeps_relative_and_absolute_floors() -> None:
    assert _continuous_wdl_gate(_wdl(), _wdl(cross_entropy=0.89)) == ()

    reasons = _continuous_wdl_gate(
        _wdl(),
        _wdl(
            cross_entropy=1.01,
            macro_cross_entropy=1.02,
            expected_score_pearson=0.15,
            loss_draw_margin=0.01,
            ece_10=0.13,
        ),
    )

    assert len(reasons) == 8


def test_rolling_policy_buffer_preserves_generation_order() -> None:
    first = PreparedTransfer(
        records=("first",),
        inputs=mx.array([[1.0]]),
        targets=mx.array([[0.75, 0.25]]),
        legal_masks=mx.array([[True, True]]),
        wdl_targets=(1,),
    )
    second = PreparedTransfer(
        records=("second",),
        inputs=mx.array([[2.0]]),
        targets=mx.array([[0.25, 0.75]]),
        legal_masks=mx.array([[True, True]]),
        wdl_targets=(0,),
    )

    combined = _combine_policy((first, second))

    assert combined.records == ("first", "second")
    assert combined.inputs.tolist() == [[1.0], [2.0]]
    assert combined.targets.tolist() == [[0.75, 0.25], [0.25, 0.75]]
    assert combined.wdl_targets == (1, 0)
