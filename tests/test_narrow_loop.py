import json
import time
from dataclasses import asdict

import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.selfplay.torch_actors import search_snapshot
from harbichess.training.torch_checkpoint import TorchCheckpointIntegrityError
from harbichess.training.torch_loop import LoopConfig, run_loop


def test_root_width_reaches_both_actor_modes_and_resume_refuses_different_width(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1015)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=2, residual_blocks=1),
        architecture="pairwise",
        invariant={"channels": 2, "blocks": 1, "hidden": 4},
    )
    config = LoopConfig(
        games=2, simulations=4, max_plies=4, steps=1, batch_size=2, workers=2, gumbel_scale=0
    )
    for threaded in (False, True):
        search, _, bridge = search_snapshot(
            network,
            {**asdict(config), "max_root_actions": 2},
            time.perf_counter() + 60,
            threaded=threaded,
        )
        assert search.config.max_considered_actions == 2
        bridge.close()
    weights = tmp_path / "initial.safetensors"
    save_weights(weights, network)
    directory = tmp_path / "run"
    run_loop(
        directory,
        config=config,
        generations=1,
        wall_seconds=60,
        weights=weights,
        max_root_actions=2,
        trainable_prefixes=("pair_",),
    )
    checkpoint = directory / "checkpoints/generation-000001"
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    assert manifest["run_config"]["max_root_actions"] == 2
    with pytest.raises(TorchCheckpointIntegrityError):
        run_loop(
            directory,
            config=config,
            generations=2,
            wall_seconds=60,
            resume=checkpoint,
            max_root_actions=4,
            trainable_prefixes=("pair_",),
        )
    resumed = run_loop(
        directory,
        config=config,
        generations=2,
        wall_seconds=60,
        resume=checkpoint,
        max_root_actions=2,
        trainable_prefixes=("pair_",),
    )
    assert resumed["state"]["generation"] == 2
    with pytest.raises(ValueError):
        run_loop(tmp_path / "invalid", config=config, generations=1, max_root_actions=0)
