from argparse import Namespace

import torch
from test_torch_resume import Uniform

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.core.state import ChessMove
from harbichess.replay.schema import records_from_game
from harbichess.replay.shard import ShardMetadata, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.selfplay.game import SelfPlayConfig, play_game
from harbichess.training.replay_distill import prepare, run


def test_distillation_masks_unknown_values_and_exactly_resumes_actual_sampler(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1212)
    weights = tmp_path / "initial.safetensors"
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
        policy_adapter={"schema": 1, "blocks": 2},
    )
    save_weights(weights, network)
    replay = tmp_path / "replay"
    rules = PythonChessRules()
    search = MCTS(Uniform(), rules=rules, config=SearchConfig(simulations=2))
    data = tuple(
        r
        for i, opening in enumerate(("e2e4", "d2d4"))
        for r in records_from_game(
            play_game(
                search,
                rules,
                rules.apply(rules.initial_state(), ChessMove(opening)),
                game_index=i,
                seed=i,
                config=SelfPlayConfig(max_plies=3),
            ),
            run_id="test",
        )
    )
    write_shard_atomic(
        replay / "data.gz",
        data,
        ShardMetadata(
            run_id="test",
            generation=1,
            source_checkpoint="test-model",
            source_commit="a" * 40,
            created_at="2026-10-03",
            split=ReplaySplit.TRAIN,
        ),
    )
    cache = tmp_path / "cache.pt"
    metadata = prepare(replay, cache)
    assert metadata["known_outcome_rows"] == {"train": 0, "validation": 0}
    panels = torch.load(cache, weights_only=True)["panels"]
    assert not panels["train"]["value_weights"].any()
    args = Namespace(
        replay=replay,
        cache=cache,
        weights=weights,
        output=tmp_path / "direct",
        resume=None,
        stop_at=4,
        wall_seconds=60,
    )
    direct = run(args)
    args.output, args.stop_at = tmp_path / "split", 2
    run(args)
    args.resume, args.stop_at = args.output / "checkpoints/step-000002", 4
    resumed = run(args)
    assert direct["state"]["sample_trace_sha256"] == resumed["state"]["sample_trace_sha256"]
    left = load_weights(tmp_path / "direct/checkpoints/step-000004/model.safetensors")
    right = load_weights(tmp_path / "split/checkpoints/step-000004/model.safetensors")
    assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
    assert resumed["state"]["baseline"]["value_ce"] is None
