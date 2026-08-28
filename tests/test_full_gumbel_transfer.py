from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.full_gumbel_transfer import (  # noqa: E402
    FullGumbelTransferConfig,
    PolicyHead,
    PolicyHeadLearner,
    _network,
    _parameter_hash,
    _policy_quality,
    _wdl_quality,
)


def test_policy_head_training_preserves_all_non_policy_parameters() -> None:
    network = _network()
    trunk = mx.random.normal((4, 8, 8, 16))
    targets = mx.zeros((4, 4_672))
    targets[:, 0] = 1.0
    masks = mx.zeros((4, 4_672), dtype=mx.bool_)
    masks[:, :2] = True
    before_frozen = _parameter_hash(network, policy=False)
    before_policy = _parameter_hash(network, policy=True)
    learner = PolicyHeadLearner(PolicyHead(network), learning_rate=2e-4)

    loss, norm = learner.train_step(trunk, targets, masks)

    assert loss > 0
    assert norm > 0
    assert _parameter_hash(network, policy=False) == before_frozen
    assert _parameter_hash(network, policy=True) != before_policy


def test_transfer_metrics_report_imitation_and_wdl_calibration() -> None:
    logits = mx.array([[2.0, 0.0], [0.0, 2.0]])
    targets = mx.array([[1.0, 0.0], [0.0, 1.0]])
    masks = mx.ones((2, 2), dtype=mx.bool_)

    policy = _policy_quality(logits, targets, masks)
    wdl = _wdl_quality(((4.0, 0.0, -1.0), (-1.0, 0.0, 4.0)), (1, -1))

    assert policy["top_action_agreement"] == 1.0
    assert policy["teacher_kl"] == pytest.approx(policy["cross_entropy"])
    assert wdl["known_positions"] == 2
    assert wdl["brier"] < 0.01


def test_transfer_config_requires_aligned_validation_steps() -> None:
    with pytest.raises(ValueError, match="align"):
        FullGumbelTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            target_result=Path("target"),
            train_shard=Path("train"),
            validation_shard=Path("validation"),
            maximum_steps=21,
            validation_interval=20,
        )
