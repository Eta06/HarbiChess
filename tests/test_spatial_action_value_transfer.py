from pathlib import Path

import chess
import pytest

from harbichess.chess.actions import move_to_action
from harbichess.training.spatial_action_value_transfer import (
    SpatialActionValueTransferConfig,
    _dense_labels,
)


def test_spatial_transfer_maps_uncertainty_labels_to_action_planes() -> None:
    board = chess.Board()
    labels = (
        ("e2e4", 0.4, 0.01, 0.75),
        ("d2d4", 0.3, 0.02, 0.25),
        ("c2c4", 0.2, 0.03, 0.0),
    )

    targets, weights, sparse = _dense_labels(board, labels)

    e4 = move_to_action(board, chess.Move.from_uci("e2e4"))
    d4 = move_to_action(board, chess.Move.from_uci("d2d4"))
    c4 = move_to_action(board, chess.Move.from_uci("c2c4"))
    assert targets[e4] == 0.4
    assert targets[d4] == 0.3
    assert weights[e4] == 0.75
    assert weights[d4] == 0.25
    assert weights[c4] == 0.0
    assert sparse == {e4: 0.4, d4: 0.3}


def test_spatial_transfer_rejects_unnormalized_uncertainty_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _dense_labels(chess.Board(), (("e2e4", 0.4, 0.01, 0.5),))


def test_spatial_transfer_configuration_freezes_checkpoints() -> None:
    config = SpatialActionValueTransferConfig(
        label_result=Path("labels.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
        architecture="move-conditioned",
    )
    assert config.checkpoint_steps == (0, 60, 120, 240, 480)
    assert config.architecture == "move-conditioned"

    with pytest.raises(ValueError, match="configuration"):
        SpatialActionValueTransferConfig(
            label_result=Path("labels.json"),
            dataset_result=Path("dataset.json"),
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            validation_shard=Path("validation.jsonl.gz"),
            output_dir=Path("output"),
            checkpoint_steps=(0, 120, 60, 480),
        )

    with pytest.raises(ValueError, match="configuration"):
        SpatialActionValueTransferConfig(
            label_result=Path("labels.json"),
            dataset_result=Path("dataset.json"),
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            validation_shard=Path("validation.jsonl.gz"),
            output_dir=Path("output"),
            architecture="unknown",  # type: ignore[arg-type]
        )
