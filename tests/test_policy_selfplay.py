"""Real opening-conditioned policy updates preserve frozen value through resume."""

import json

import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.replay.shard import read_shard
from harbichess.training.torch_loop import LoopConfig, run_loop


@pytest.mark.parametrize("actor_mode", ("thread", "process"))
def test_real_opening_policy_learning_freeze_and_resume(tmp_path, actor_mode):
    torch.set_num_threads(1)
    torch.manual_seed(298)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    weights = tmp_path / "initial.safetensors"
    save_weights(weights, network)
    opening = ["e2e4", "e7e5", "g1f3", "b8c6"]
    book = tmp_path / "openings.json"
    book.write_text(
        json.dumps({"schema": 1, "splits": {"train": [{"opening": {"moves": opening}}]}})
    )
    run = tmp_path / "run"
    config = LoopConfig(games=2, simulations=2, max_plies=8, steps=2, batch_size=2, workers=1)
    first = run_loop(
        run,
        config=config,
        generations=1,
        weights=weights,
        actor_mode=actor_mode,
        trainable_prefixes=("pair_",),
        opening_book=book,
    )
    assert first["status"] == "completed"
    for shard in (run / "replay").glob("*.gz"):
        data = read_shard(shard)
        assert data.records and all(
            tuple(opening) == r.moves[:4] and r.ply >= 4 for r in data.records
        )
    checkpoint = run / first["last_complete_checkpoint"]
    selected = load_weights(checkpoint / "model.safetensors")
    assert any(
        not torch.equal(v, selected.state_dict()[k])
        for k, v in network.state_dict().items()
        if k.startswith("pair_")
    )
    assert all(
        torch.equal(v, selected.state_dict()[k])
        for k, v in network.state_dict().items()
        if not k.startswith("pair_")
    )
    second = run_loop(
        run,
        config=config,
        generations=2,
        resume=checkpoint,
        actor_mode=actor_mode,
        trainable_prefixes=("pair_",),
        opening_book=book,
    )
    final = load_weights(run / second["last_complete_checkpoint"] / "model.safetensors")
    assert all(
        torch.equal(v, final.state_dict()[k])
        for k, v in network.state_dict().items()
        if not k.startswith("pair_")
    )
    manifest = json.loads(
        (run / second["last_complete_checkpoint"] / "checkpoint.json").read_text()
    )
    assert manifest["trainable"] and all(k.startswith("pair_") for k in manifest["trainable"])
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_loop(
            run,
            config=config,
            generations=3,
            resume=run / second["last_complete_checkpoint"],
            opening_book=book,
            actor_mode=actor_mode,
        )
    book.write_text(
        json.dumps({"schema": 1, "splits": {"train": [{"opening": {"moves": ["d2d4", "d7d5"]}}]}})
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_loop(
            run,
            config=config,
            generations=3,
            resume=run / second["last_complete_checkpoint"],
            actor_mode=actor_mode,
            trainable_prefixes=("pair_",),
            opening_book=book,
        )
