import chess
import torch

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.evaluation.portable_arena import arena
from harbichess.search.full_gumbel import FullGumbelMCTS


def test_policy_only_arena_plays_real_legal_moves_without_search(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    weights = tmp_path / "model.safetensors"
    save_weights(weights, TorchChessNetwork(NetworkConfig(trunk_channels=2, residual_blocks=1)))

    def refused_search(*args, **kwargs):
        raise AssertionError("policy-only candidate must not run search")

    monkeypatch.setattr(FullGumbelMCTS, "search", refused_search)
    result = arena(
        weights,
        opponent="random",
        candidate_policy_only=True,
        simulations=16,
        max_plies=4,
        opening_pairs=1,
        openings=(("e2e4", "c7c5"),),
    )
    assert result["candidate_selection"] == "raw_policy_argmax"
    assert result["candidate_neural_simulations_per_move"] == 0
    assert result["opponent_neural_simulations_per_move"] is None
    assert result["evaluated_positions"] == 2  # Exactly one evaluation per candidate move.
    assert result["summary"]["capped_games"] == 2
    for game in result["games"]:
        board = chess.Board()
        for uci in game["moves"]:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
