from pathlib import Path

import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.training.capacity_matrix import (
    FROZEN_VARIANTS,
    CapacityMatrixConfig,
    _gate_reasons,
    _tactical_metrics,
)


def _row(*, cross_entropy: float = 2.5, top: float = 0.45, tactical=(1, 6)):
    return {
        "initial_logit_delta": 0.0,
        "gradients_finite": True,
        "checkpoints": [
            {
                "quality": {
                    "teacher_policy_cross_entropy": cross_entropy,
                    "teacher_top_action_agreement": top,
                }
            }
        ],
        "tactical": {
            "raw": {"solved": tactical[0]},
            "budgets": [{"solved": tactical[1]}],
        },
    }


def test_capacity_matrix_freezes_the_preregistered_architectures() -> None:
    assert tuple(
        (variant.name, variant.residual_blocks, variant.policy_channels)
        for variant in FROZEN_VARIANTS
    ) == (
        ("base", 2, 4),
        ("deep", 4, 4),
        ("head", 2, 8),
        ("deep-head", 4, 8),
    )


def test_capacity_gate_requires_quality_and_tactical_non_regression() -> None:
    config = CapacityMatrixConfig(Path("replay.json"), Path("output"))
    base = {
        "teacher_policy_cross_entropy": 2.6,
        "teacher_top_action_agreement": 0.30,
    }

    assert not _gate_reasons(
        _row(),
        base_final=base,
        baseline_top=0.407067,
        baseline_tactical=(1, 6),
        config=config,
    )
    assert "teacher top-action agreement did not beat release baseline" in _gate_reasons(
        _row(top=0.40),
        base_final=base,
        baseline_top=0.407067,
        baseline_tactical=(1, 6),
        config=config,
    )
    assert "tactical solve count regressed" in _gate_reasons(
        _row(tactical=(1, 5)),
        base_final=base,
        baseline_top=0.407067,
        baseline_tactical=(1, 6),
        config=config,
    )


def test_capacity_matrix_configuration_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="configuration"):
        CapacityMatrixConfig(Path("replay.json"), Path("output"), epochs=0)


def test_capacity_matrix_tactical_smoke_uses_one_rules_engine() -> None:
    network = HarbiChessNetwork(
        NetworkConfig(trunk_channels=8, residual_blocks=1, policy_channels=2)
    )

    result = _tactical_metrics(network, budget=1, workers=1, seed=7)

    assert result["raw"]["total"] == 8
    assert result["budgets"][0]["budget"] == 1
