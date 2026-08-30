from pathlib import Path

import mlx.core as mx
import pytest

from harbichess.evaluation.invariant_value_probe import (
    InvariantValueProbeConfig,
    _material_soft_targets,
    _select_arm,
)


def _arm(*, passed: bool, mse: float) -> dict[str, object]:
    return {"passed": passed, "selected": {"mse": mse}}


def test_selection_requires_a_passing_arm_and_prefers_simplicity() -> None:
    assert (
        _select_arm(
            {
                "global-linear": _arm(passed=False, mse=0.01),
                "invariant-tower": _arm(passed=False, mse=0.001),
            }
        )
        is None
    )
    assert (
        _select_arm(
            {
                "global-linear": _arm(passed=True, mse=0.01),
                "invariant-tower": _arm(passed=True, mse=0.009),
            }
        )
        == "global-linear"
    )


def test_selection_requires_twenty_percent_tower_mse_gain() -> None:
    arms = {
        "global-linear": _arm(passed=True, mse=0.01),
        "invariant-tower": _arm(passed=True, mse=0.008),
    }

    assert _select_arm(arms) == "invariant-tower"


def test_probe_config_rejects_unaligned_schedule() -> None:
    with pytest.raises(ValueError, match="configuration"):
        InvariantValueProbeConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            steps=21,
            validation_interval=20,
        )


def test_material_soft_targets_identify_draw_probability() -> None:
    targets = _material_soft_targets(mx.array((-0.25, 0.0, 0.40)))
    mx.eval(targets)

    rows = targets.tolist()
    for actual, expected in zip(
        rows,
        ([0.0, 0.75, 0.25], [0.0, 1.0, 0.0], [0.40, 0.60, 0.0]),
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert [sum(row) for row in rows] == pytest.approx([1.0, 1.0, 1.0])
