import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.chess.actions import move_to_action
from harbichess.core.network_config import NetworkConfig
from harbichess.training.oracle_data import publish_json
from harbichess.training.oracle_train import run


def test_real_soft_target_updates_resume_with_identical_optimizer_and_rng(tmp_path):
    import chess

    dataset = tmp_path / "data"
    dataset.mkdir()
    files = {}
    for split, opening in (("train", "e2e4"), ("validation", "d2d4")):
        board = chess.Board()
        board.push_uci(opening)
        move = chess.Move.from_uci("e7e5")
        row = {
            "root_fen": chess.STARTING_FEN,
            "moves": [opening],
            "fen": board.fen(),
            "position_key": " ".join(board.fen().split()[:4]),
            "legal": [[m.uci(), move_to_action(board, m)] for m in board.legal_moves],
            "policy": [[move.uci(), move_to_action(board, move), 1.0]],
            "wdl": [0.3, 0.6, 0.1],
            "family": 0 if split == "train" else 1,
            "split": split,
        }
        path = dataset / f"{split}.json.gz"
        publish_json(
            path,
            {
                "schema": 1,
                "target_semantics": "engine-reference",
                "job": {"split": split, "family": row["family"]},
                "rows": [row],
            },
        )
        files[path.name] = sha256(path)
    publish_json(dataset / "dataset.json", {"files": files})
    publish_json(dataset / "metadata.json", {"schema": 1})
    torch.manual_seed(123)
    network = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=2, residual_blocks=1, policy_channels=1, value_channels=1, value_hidden=2
        )
    )
    weights = tmp_path / "initial.safetensors"
    save_weights(weights, network)
    config = dict(max_steps=4, interval=1, patience=10, batch_size=2)
    straight, restarted = tmp_path / "straight", tmp_path / "restarted"
    run(straight, dataset, weights, **config)
    paused = run(restarted, dataset, weights, stop_at=2, **config)
    assert paused["reason"] == "registered process boundary"
    resumed = run(
        restarted, dataset, weights, resume=restarted / "checkpoints/step-000002", **config
    )
    assert resumed["step"] == 4
    a = straight / "checkpoints/step-000004"
    b = restarted / "checkpoints/step-000004"
    for key, value in load_weights(a / "model.safetensors").state_dict().items():
        assert torch.equal(value, load_weights(b / "model.safetensors").state_dict()[key])
    x = torch.load(a / "training.pt", weights_only=True)
    y = torch.load(b / "training.pt", weights_only=True)
    assert torch.equal(x["torch_rng"], y["torch_rng"])
    for i, state in x["optimizer"]["state"].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, y["optimizer"]["state"][i][key])
    assert any(
        not torch.equal(p, network.state_dict()[k])
        for k, p in load_weights(a / "model.safetensors").state_dict().items()
    )
