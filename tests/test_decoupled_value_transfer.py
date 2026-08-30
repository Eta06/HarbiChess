from pathlib import Path
from types import SimpleNamespace

import pytest

from harbichess.training.decoupled_value_transfer import (
    DecoupledValueTransferConfig,
    _material_reasons,
    _MixedWDLSampler,
    _select_wdl_arm,
)


def test_auxiliary_material_gate_keeps_original_thresholds() -> None:
    baseline = {"mse": 0.02, "mae": 0.11, "pearson": 0.0}

    assert _material_reasons(baseline, {"mse": 0.001, "mae": 0.01, "pearson": 0.99}) == ()
    assert len(_material_reasons(baseline, {"mse": 0.02, "mae": 0.06, "pearson": 0.79})) == 3


def test_wdl_selection_rejects_failed_arms_and_uses_macro_ce() -> None:
    assert (
        _select_wdl_arm(
            {
                "global-wdl": {"passed": False, "selected_macro_wdl_ce": 0.1},
                "global-tower-wdl": {
                    "passed": False,
                    "selected_macro_wdl_ce": 0.05,
                },
            }
        )
        is None
    )
    assert (
        _select_wdl_arm(
            {
                "global-wdl": {"passed": True, "selected_macro_wdl_ce": 1.0},
                "global-tower-wdl": {
                    "passed": True,
                    "selected_macro_wdl_ce": 0.9,
                },
            }
        )
        == "global-tower-wdl"
    )


def test_config_rejects_unaligned_stages() -> None:
    with pytest.raises(ValueError, match="configuration"):
        DecoupledValueTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            wdl_steps=21,
            validation_interval=20,
        )


def test_config_rejects_unknown_wdl_sampling_mode() -> None:
    with pytest.raises(ValueError, match="sampling mode"):
        DecoupledValueTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            wdl_sampling_mode="adaptive",
        )


def test_mixed_wdl_sampler_is_deterministic_and_keeps_batch_size() -> None:
    records = tuple(
        SimpleNamespace(
            game_id=f"game-{outcome}-{game}",
            outcome_value=outcome,
            repetition_redirected=False,
        )
        for outcome in (-1, 0, 1)
        for game in range(3)
    )

    left = _MixedWDLSampler(records, seed=17).sample_indices(64)
    right = _MixedWDLSampler(records, seed=17).sample_indices(64)

    assert left == right
    assert len(left) == 64
    assert all(0 <= index < len(records) for index in left)
