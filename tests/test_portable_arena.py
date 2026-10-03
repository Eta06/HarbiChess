import chess
import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.evaluation.portable_arena import arena


def test_frozen_suite_preserves_histories_colours_and_move_times(tmp_path):
    torch.set_num_threads(1)
    weights = tmp_path / "model.safetensors"
    save_weights(weights, TorchChessNetwork(NetworkConfig(trunk_channels=2, residual_blocks=1)))
    openings = (("e2e4", "c7c5"), ("d2d4", "d7d5"))
    result = arena(
        weights,
        opponent="random",
        simulations=2,
        max_plies=4,
        opening_pairs=2,
        openings=openings,
        opening_source_sha256="a" * 64,
    )
    assert len(result["games"]) == 4
    assert result["opening_source_sha256"] == "a" * 64
    for i, game in enumerate(result["games"]):
        assert game["candidate_color"] == ("white" if i % 2 == 0 else "black")
        assert game["moves"][:2] == list(openings[i // 2])
        assert len(game["move_wall_seconds"]) == len(game["moves"]) - 2
        assert all(t >= 0 for t in game["move_wall_seconds"])
        board = chess.Board()
        for uci in game["moves"]:
            board.push_uci(uci)
    # Unknown capped outcomes remain explicitly identified, not terminal draws.
    assert result["summary"]["capped_games"] == 4
    with pytest.raises(ValueError):
        arena(weights, opponent="random", openings=(("e2e5",),), opening_pairs=1)
