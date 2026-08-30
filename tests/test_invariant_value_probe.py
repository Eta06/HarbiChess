from pathlib import Path

import pytest

from harbichess.evaluation.invariant_value_probe import (
    InvariantValueProbeConfig,
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
