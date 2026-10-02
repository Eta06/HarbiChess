"""Separate-process optimizer/RNG and full CLI generation resume regression tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.selfplay.game import SelfPlayConfig, play_game
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import (
    TorchCheckpointIntegrityError,
    load_checkpoint,
    save_checkpoint,
)
from harbichess.training.torch_learner import TorchLearner


class Uniform:
    def evaluate(self, state):
        moves = PythonChessRules().legal_moves(state)
        return PositionEvaluation(tuple((move, 1 / len(moves)) for move in moves), 0.0)


def records():
    rules = PythonChessRules()
    search = MCTS(Uniform(), rules=rules, config=SearchConfig(simulations=2))
    return tuple(
        r
        for i in range(2)
        for r in records_from_game(
            play_game(
                search,
                rules,
                rules.initial_state(),
                game_index=i,
                seed=i,
                config=SelfPlayConfig(max_plies=3),
            ),
            run_id="test",
        )
    )


def test_separate_process_exact_optimizer_sampler_rng_resume(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(91)
    learner = TorchLearner(
        TorchChessNetwork(NetworkConfig(trunk_channels=4, residual_blocks=1)),
        config=LearnerConfig(learning_rate=0.001),
    )
    data = records()
    prepared = learner.prepare_batch(build_training_batch(data))
    sampler = GameBalancedSampler(data, seed=92)
    learner.train_step(prepared.select(sampler.sample_indices(3)))
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        learner=learner,
        sampler=sampler,
        replay_paths=(),
        run_state={"generation": 1},
        run_config={"seed": 92},
        source_commit="a" * 40,
    )
    indices = sampler.sample_indices(3)
    metrics = learner.train_step(prepared.select(indices))
    expected_rng = torch.rand(3)
    worker = """
import json, sys, torch
from pathlib import Path
from harbichess.training.torch_checkpoint import load_checkpoint
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from test_torch_resume import records
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
learner, manifest, rng = load_checkpoint(Path(sys.argv[1]), expected_run_config={"seed":92})
data=records()
sampler=GameBalancedSampler(data, seed=0)
sampler.set_rng_state(rng)
indices=sampler.sample_indices(3)
metrics=learner.train_step(learner.prepare_batch(build_training_batch(data)).select(indices))
torch.save({"weights":learner.network.state_dict(),"indices":indices,
            "loss":metrics.total_loss,"rng":torch.rand(3)},sys.argv[2])
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
    output = tmp_path / "resumed.pt"
    subprocess.run(
        [sys.executable, "-c", worker, str(checkpoint), str(output)],
        check=True,
        env=environment,
        timeout=60,
    )
    resumed = torch.load(output, weights_only=True)
    assert resumed["indices"] == indices
    assert resumed["loss"] == metrics.total_loss
    assert torch.equal(resumed["rng"], expected_rng)
    assert all(
        torch.equal(v, resumed["weights"][k]) for k, v in learner.network.state_dict().items()
    )
    with pytest.raises(TorchCheckpointIntegrityError, match="configuration"):
        load_checkpoint(checkpoint, expected_run_config={"seed": 93})
    with (checkpoint / "training.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(TorchCheckpointIntegrityError, match="checksum"):
        load_checkpoint(checkpoint, expected_run_config={"seed": 92})


def test_cli_generation_resume_matches_uninterrupted_run(tmp_path):
    # Same run basename is part of replay game identity; different parent directories isolate runs.
    full, resumed = tmp_path / "full/run", tmp_path / "resumed/run"
    full.parent.mkdir()
    resumed.parent.mkdir()
    options = [
        "--games",
        "2",
        "--simulations",
        "2",
        "--max-plies",
        "4",
        "--steps",
        "2",
        "--batch-size",
        "2",
        "--workers",
        "1",
        "--threads",
        "1",
        "--wall-seconds",
        "60",
    ]

    def run(directory, target, resume=None):
        command = [
            sys.executable,
            "-m",
            "harbichess.training.torch_loop",
            str(directory),
            "--generations",
            str(target),
            *options,
        ]
        if resume:
            command += ["--resume", str(resume)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)

    run(full, 2)
    run(resumed, 1)
    run(resumed, 2, resumed / "checkpoints/generation-000001")
    a = load_file(str(full / "checkpoints/generation-000002/model.safetensors"))
    b = load_file(str(resumed / "checkpoints/generation-000002/model.safetensors"))
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    result = json.loads((resumed / "result.json").read_text())
    assert result["state"]["generation"] == 2 and result["state"]["next_game"] == 4
    assert result["status"] == "completed"
    assert len(result["state"]["history"]) == 2
    assert result["state"]["history"][0]["terminal_observed_rows"] == 0
    assert result["state"]["history"][1]["metrics"][-1]["step"] == 4
    # Mutating replay must be detected before resumed execution.
    replay = next((resumed / "replay").glob("*.gz"))
    with replay.open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(TorchCheckpointIntegrityError, match="checksum"):
        load_checkpoint(
            resumed / "checkpoints/generation-000002", expected_run_config=result["config"]
        )
