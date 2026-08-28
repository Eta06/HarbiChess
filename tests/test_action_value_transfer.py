from pathlib import Path

import mlx.core as mx

from harbichess.backends.action_value_network import HarbiChessActionValueNetwork
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS
from harbichess.training.action_value_transfer import (
    ActionValueLearner,
    ActionValueTransferConfig,
    _gate_reasons,
)


def _config() -> ActionValueTransferConfig:
    return ActionValueTransferConfig(
        q_reliability_result=Path("q.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_action_value_learner_updates_only_new_head() -> None:
    mx.random.seed(73)
    config = NetworkConfig(trunk_channels=8, residual_blocks=1, value_channels=2)
    base = HarbiChessNetwork(config)
    network = HarbiChessActionValueNetwork.from_base(base, action_value_channels=2)
    inputs = mx.random.uniform(shape=(2, 8, 8, ENCODER_CHANNELS))
    trunk, state_values = network.frozen_action_features(inputs)
    targets = mx.zeros((2, POLICY_SIZE))
    targets[:, 0] = 0.8
    weights = mx.zeros((2, POLICY_SIZE))
    weights[:, 0] = 1.0
    before_policy, before_wdl, before_q = network.forward_with_action_values(inputs)
    mx.eval(before_policy, before_wdl, before_q)

    learner = ActionValueLearner(network, learning_rate=2e-4, max_gradient_norm=5.0)
    for _ in range(3):
        learner.train_step((trunk, state_values, targets, weights))
    after_policy, after_wdl, after_q = network.forward_with_action_values(inputs)
    mx.eval(after_policy, after_wdl, after_q)

    assert float(mx.max(mx.abs(after_policy - before_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(after_wdl - before_wdl)).item()) == 0.0
    assert float(mx.max(mx.abs(after_q - before_q)).item()) > 0.0


def test_action_value_gate_requires_transfer_strength_and_invariance() -> None:
    quality = {
        "weighted_q_mse": 0.07,
        "mean_teacher_q_spearman": 0.40,
        "verified_delta_95_interval": (0.01, 0.10),
        "harmful_ratio": 0.08,
        "mean_verified_regret": 0.09,
        "best_action_coverage_top_16": 0.82,
        "maximum_policy_wdl_logit_delta": 0.0,
    }
    assert _gate_reasons(
        quality,
        baseline_mse=0.10,
        config=_config(),
        tactical=(1, 6),
        baseline_tactical=(1, 6),
        maximum_gradient_norm=2.0,
        maximum_unclipped_gradient_norm=2.0,
    ) == ()

    failed = {**quality, "weighted_q_mse": 0.09, "best_action_coverage_top_16": 0.70}
    assert _gate_reasons(
        failed,
        baseline_mse=0.10,
        config=_config(),
        tactical=(1, 5),
        baseline_tactical=(1, 6),
        maximum_gradient_norm=2.0,
        maximum_unclipped_gradient_norm=2.0,
    ) == (
        "validation Q MSE did not improve by 20%",
        "predicted-Q top-16 best-action coverage is below 80%",
        "release tactical solve counts changed",
    )
