import json
from types import SimpleNamespace

import chess
import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.evaluation.portable_arena import arena


def test_actual_root_widths_and_durable_move_trace(tmp_path):
    torch.set_num_threads(1)
    weights = tmp_path / "model.safetensors"
    save_weights(weights, TorchChessNetwork(NetworkConfig(trunk_channels=2, residual_blocks=1)))
    trace = tmp_path / "progress.jsonl"
    result = arena(
        weights,
        opponent=weights,
        simulations=4,
        max_plies=4,
        openings=(("e2e4", "e7e5"),),
        opening_pairs=1,
        candidate_root_actions=2,
        opponent_root_actions=4,
        progress_path=trace,
    )
    assert result["candidate_root_actions"] == 2 and result["opponent_root_actions"] == 4
    events = [json.loads(s) for s in trace.read_text().splitlines()]
    assert [e["game"] for e in events if e["type"] == "game_end"] == json.loads(
        json.dumps(result["games"])
    )
    assert len([e for e in events if e["type"] == "move"]) == 4
    with pytest.raises(ValueError):
        arena(weights, opponent="random", candidate_root_actions=0)
    with pytest.raises(FileExistsError):
        arena(weights, opponent="random", progress_path=trace)


def test_records_engine_actual_nodes_and_requests_basic_info(tmp_path, monkeypatch):
    class Engine:
        def __init__(self):
            self.id = {"name": "test"}

        def configure(self, options):
            assert options == {"Threads": 1, "Hash": 16}

        def play(self, board, limit, *, game, info):
            assert limit.nodes == 32 and info == chess.engine.INFO_BASIC
            return SimpleNamespace(
                move=sorted(board.legal_moves, key=lambda m: m.uci())[0], info={"nodes": 34}
            )

        def quit(self):
            pass

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **kw: Engine())
    weights, engine_path = tmp_path / "model.safetensors", tmp_path / "engine"
    engine_path.write_text("test")
    save_weights(weights, TorchChessNetwork(NetworkConfig(trunk_channels=2, residual_blocks=1)))
    result = arena(
        weights,
        opponent="stockfish",
        stockfish=engine_path,
        simulations=2,
        max_plies=4,
        openings=(("e2e4", "e7e5"),),
        opening_pairs=1,
        record_engine_nodes=True,
    )
    assert result["stockfish_actual_nodes"] == 68
    assert all(len(g["stockfish_nodes_by_move"]) == 1 for g in result["games"])
