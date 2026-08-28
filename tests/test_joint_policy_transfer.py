from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.encoding import ENCODER_CHANNELS
from harbichess.training.joint_policy_transfer import (
    JointPolicyLearner,
    JointPolicyTransferConfig,
    _joint_reasons,
    _masked_kl,
    _target_arrays,
)


def _config(tmp_path: Path) -> JointPolicyTransferConfig:
    return JointPolicyTransferConfig(
        policy_target_result=tmp_path / "target.json",
        dataset_result=tmp_path / "dataset.json",
        run_result=tmp_path / "run.json",
        train_shard=tmp_path / "train.jsonl.gz",
        output_dir=tmp_path / "output",
    )


def test_frozen_joint_matrix_matches_preregistration(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.steps == 960
    assert config.target_batch_size == 16
    assert config.anchor_batch_size == 64
    assert config.learning_rate == 2e-4
    assert config.policy_anchor_weights == (1.0, 4.0, 16.0)
    assert config.wdl_anchor_weight == 4.0
    assert config.maximum_anchor_kl == 0.02
    assert config.maximum_wdl_anchor_kl == 0.002
    assert config.maximum_expected_score_drift == 0.02


def test_joint_gate_adds_wdl_and_expected_score_guards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    quality = {
        "mean_teacher_policy_spearman": 0.5,
        "verified_delta_95_interval": (0.01, 0.04),
        "harmful_ratio": 0.08,
        "mean_verified_regret": 0.08,
        "best_action_coverage_top_16": 0.9,
    }

    assert not _joint_reasons(
        quality,
        gap_fraction=0.25,
        policy_kl=0.01,
        wdl_kl=0.001,
        expected_score_drift=0.01,
        maximum_gradient_norm=4.0,
        config=config,
    )
    assert _joint_reasons(
        quality,
        gap_fraction=0.25,
        policy_kl=0.01,
        wdl_kl=0.003,
        expected_score_drift=0.03,
        maximum_gradient_norm=4.0,
        config=config,
    ) == (
        "broad replay WDL KL exceeds 0.002",
        "broad replay expected-score drift exceeds 0.02",
    )


def test_joint_learner_accepts_soft_targets_and_distillation() -> None:
    mx.random.seed(7)
    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=2,
            residual_blocks=1,
            policy_channels=1,
            value_channels=1,
            value_hidden=2,
        )
    )
    inputs = mx.zeros((2, 8, 8, ENCODER_CHANNELS))
    base_policy, base_wdl = network(inputs)
    masks = mx.zeros_like(base_policy, dtype=mx.bool_)
    masks[:, :2] = True
    targets = mx.zeros_like(base_policy)
    targets[:, 0] = 0.6
    targets[:, 1] = 0.4
    learner = JointPolicyLearner(
        network,
        learning_rate=2e-4,
        policy_anchor_weight=4.0,
        wdl_anchor_weight=4.0,
        max_gradient_norm=5.0,
    )

    loss, gradient_norm = learner.train_step(
        (inputs, targets, masks, inputs, base_policy, base_wdl, masks)
    )

    assert loss > 0
    assert gradient_norm > 0
    candidate_policy, candidate_wdl = network(inputs)
    assert _masked_kl(base_policy, candidate_policy, masks) >= 0
    assert _masked_kl(base_wdl, candidate_wdl) >= 0


def test_target_batch_selects_board_inputs_not_frozen_features() -> None:
    data = SimpleNamespace(
        inputs=mx.arange(24).reshape(3, 2, 2, 2),
        targets=mx.arange(12).reshape(3, 4),
        legal_masks=mx.ones((3, 4), dtype=mx.bool_),
    )

    inputs, targets, masks = _target_arrays(data, (2, 0))  # type: ignore[arg-type]

    assert tuple(inputs.shape) == (2, 2, 2, 2)
    assert inputs[:, 0, 0, 0].tolist() == [16, 0]
    assert targets[:, 0].tolist() == [8, 0]
    assert masks.shape == targets.shape
