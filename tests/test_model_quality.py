import math

import pytest
from test_replay_schema import scripted_game

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.evaluation.model_quality import evaluate_model_quality
from harbichess.replay.schema import records_from_game

pytest.importorskip("mlx.core")


def test_model_quality_reports_policy_and_calibration_metrics() -> None:
    rules, game = scripted_game()
    records = records_from_game(game, run_id="quality", rules=rules)
    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=1,
            value_hidden=8,
        )
    )

    metrics = evaluate_model_quality(network, records, batch_size=2, rules=rules)

    assert metrics.samples == metrics.known_value_samples == 4
    assert math.isfinite(metrics.teacher_policy_cross_entropy)
    assert 0.0 <= metrics.teacher_top_action_agreement <= 1.0
    assert math.isfinite(metrics.value_cross_entropy)
    assert 0.0 <= metrics.value_accuracy <= 1.0
    assert 0.0 <= metrics.expected_score_ece <= 1.0
    assert 0.0 <= metrics.expected_score_brier <= 1.0


def test_model_quality_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="requires records"):
        evaluate_model_quality(HarbiChessNetwork(), ())
