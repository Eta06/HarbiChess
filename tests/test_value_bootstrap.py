from pathlib import Path

import pytest
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.training.value_bootstrap import (
    ValueBootstrapConfig,
    _freeze_to_value_head,
)


def test_value_bootstrap_freezes_policy_and_trunk() -> None:
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))

    _freeze_to_value_head(network)

    trainable = {name for name, _ in tree_flatten(network.trainable_parameters())}
    assert trainable
    assert all(name.startswith("value_") for name in trainable)
    assert "value_output.weight" in trainable


def test_value_bootstrap_requires_aligned_validation_intervals() -> None:
    with pytest.raises(ValueError, match="configuration"):
        ValueBootstrapConfig(
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            validation_shard=Path("validation.jsonl.gz"),
            output_dir=Path("diagnostics"),
            steps=10,
            validation_interval=6,
        )
