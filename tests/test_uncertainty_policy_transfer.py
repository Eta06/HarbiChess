from pathlib import Path

import chess
import mlx.core as mx
import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.training.uncertainty_policy_transfer import (
    LowRankPolicyAdapter,
    UncertaintyPolicyTransferConfig,
    _dense_explicit_target,
    _dense_target,
)


def _config() -> UncertaintyPolicyTransferConfig:
    return UncertaintyPolicyTransferConfig(
        label_result=Path("labels.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_uncertainty_policy_target_preserves_soft_mass_and_legality() -> None:
    board = chess.Board()
    labels = (
        ("e2e4", 0.4, 0.01, 0.7),
        ("d2d4", 0.3, 0.02, 0.3),
        ("c2c4", 0.2, 0.04, 0.0),
    )

    target, legal_mask, teacher, legal = _dense_target(board, labels)

    e4 = move_to_action(board, chess.Move.from_uci("e2e4"))
    d4 = move_to_action(board, chess.Move.from_uci("d2d4"))
    c4 = move_to_action(board, chess.Move.from_uci("c2c4"))
    assert len(target) == len(legal_mask) == POLICY_SIZE
    assert sum(target) == pytest.approx(1.0)
    assert target[e4] == pytest.approx(0.7)
    assert target[d4] == pytest.approx(0.3)
    assert c4 not in teacher
    assert all(legal_mask[action] for action in legal)


def test_low_rank_adapter_is_function_preserving_and_mergeable() -> None:
    mx.random.seed(91)
    config = NetworkConfig(trunk_channels=16, residual_blocks=1, policy_channels=2)
    network = HarbiChessNetwork(config)
    features = mx.random.uniform(shape=(3, 128))
    base_logits = network.policy_linear(features)
    adapter = LowRankPolicyAdapter(128, 4)

    adapted = adapter(features, base_logits)
    merged = features @ adapter.merged_weight(network.policy_linear.weight).T
    merged = merged + network.policy_linear.bias
    mx.eval(base_logits, adapted, merged)

    assert float(mx.max(mx.abs(adapted - base_logits)).item()) == 0.0
    assert float(mx.max(mx.abs(merged - base_logits)).item()) == 0.0

    adapter.up.weight = mx.ones_like(adapter.up.weight) * 0.01
    full = adapter(features, base_logits)
    half = base_logits + 0.5 * (full - base_logits)
    mx.eval(full, half)
    assert float(mx.max(mx.abs(full - base_logits)).item()) > 0.0
    error = mx.max(mx.abs((half - base_logits) - 0.5 * (full - base_logits)))
    mx.eval(error)
    assert float(error.item()) < 1e-7


def test_explicit_policy_target_maps_uci_probabilities() -> None:
    board = chess.Board()
    target, legal_mask, teacher, _legal = _dense_explicit_target(
        board, (("e2e4", 0.6), ("d2d4", 0.4))
    )
    e4 = move_to_action(board, chess.Move.from_uci("e2e4"))
    d4 = move_to_action(board, chess.Move.from_uci("d2d4"))
    assert target[e4] == pytest.approx(0.6)
    assert target[d4] == pytest.approx(0.4)
    assert legal_mask[e4] and legal_mask[d4]
    assert teacher == {e4: 0.6, d4: 0.4}


def test_uncertainty_policy_config_rejects_unfrozen_checkpoints() -> None:
    assert _config().checkpoint_steps == (0, 60, 120, 240, 480)
    with pytest.raises(ValueError, match="configuration"):
        UncertaintyPolicyTransferConfig(
            label_result=Path("labels.json"),
            dataset_result=Path("dataset.json"),
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            validation_shard=Path("validation.jsonl.gz"),
            output_dir=Path("output"),
            checkpoint_steps=(0, 120, 60, 480),
        )
