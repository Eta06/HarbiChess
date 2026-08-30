from pathlib import Path

import pytest

from harbichess.training.invariant_wdl_transfer import (
    InvariantWDLTransferConfig,
    _material_gate_reasons,
    _select_arm,
)


def _arm(*, passed: bool, macro: float) -> dict[str, object]:
    return {"passed": passed, "selected_macro_wdl_ce": macro}


def test_material_gate_requires_all_frozen_thresholds() -> None:
    release = {"mse": 0.02, "mae": 0.11, "pearson": 0.0}

    assert _material_gate_reasons(
        release, {"mse": 0.001, "mae": 0.01, "pearson": 0.99}
    ) == ()
    assert len(
        _material_gate_reasons(
            release, {"mse": 0.02, "mae": 0.06, "pearson": 0.79}
        )
    ) == 3


def test_selection_prefers_plain_arm_within_ce_tolerance() -> None:
    assert (
        _select_arm(
            {
                "tower-wdl": _arm(passed=True, macro=1.00),
                "tower-wdl-retained": _arm(passed=True, macro=0.995),
            }
        )
        == "tower-wdl"
    )
    assert (
        _select_arm(
            {
                "tower-wdl": _arm(passed=True, macro=1.00),
                "tower-wdl-retained": _arm(passed=True, macro=0.98),
            }
        )
        == "tower-wdl-retained"
    )


def test_selection_rejects_failed_arms() -> None:
    assert (
        _select_arm(
            {
                "tower-wdl": _arm(passed=False, macro=0.5),
                "tower-wdl-retained": _arm(passed=False, macro=0.4),
            }
        )
        is None
    )


def test_config_rejects_unaligned_schedule() -> None:
    with pytest.raises(ValueError, match="configuration"):
        InvariantWDLTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            material_result=Path("result"),
            steps=21,
            validation_interval=20,
        )
