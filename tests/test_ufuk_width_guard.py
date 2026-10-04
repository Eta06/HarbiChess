import chess
import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig
from harbichess.training import oracle_train
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch
from harbichess.training.ufuk_width_guard import guarded


def test_retention_stop_saves_current_complete_state_and_refuses_more_updates(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    board = chess.Board()
    inputs = torch.tensor(BoardEncoder().encode_board(board).values).reshape(1, 8, 8, 104)
    masks = torch.zeros(1, 4672, dtype=torch.bool)
    actions = legal_action_indices(board)
    masks[:, actions] = True
    policy = torch.zeros(1, 4672)
    policy[0, actions[0]] = 1
    batch = TorchTrainingBatch(inputs, policy, masks, torch.tensor([0]), torch.ones(1))
    weights = tmp_path / "initial.safetensors"
    save_weights(weights, TorchChessNetwork(NetworkConfig(trunk_channels=4, residual_blocks=1)))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}\n")
    directory = tmp_path / "run"
    captured = {}

    def measurement(learner, panel):
        # Force a validation boundary while keeping actual updates/native state real.
        return {
            "policy_ce": 2.0,
            "value_ce": 0.5 if learner.step == 0 else 0.55,
            "value_mae": 0.2,
            "total_ce": 2.5,
        }

    def producer(directory, dataset, weights, **kwargs):
        directory.mkdir()
        learner = TorchLearner(load_weights(weights), config=LearnerConfig(learning_rate=5e-5))
        measured = oracle_train.evaluate(learner, batch)
        save_checkpoint(
            directory / "checkpoints/step-000000",
            learner=learner,
            sampler=None,
            replay_paths=(dataset / "dataset.json",),
            run_state={
                "cursor": 0,
                "best_step": 0,
                "best_score": 2.5,
                "evaluations": [{"step": 0, **measured}],
            },
            run_config={"fixture": "current-state-retention"},
            source_commit="0" * 40,
        )
        learner.train_step(batch)
        learner.train_step(batch)
        captured["learner"] = learner
        captured["rng"] = torch.get_rng_state().clone()
        oracle_train.evaluate(learner, batch)
        pytest.fail("invalid native retention must stop producer")

    monkeypatch.setattr(oracle_train, "evaluate", measurement)
    monkeypatch.setattr(oracle_train, "run", producer)
    result = guarded(directory, dataset, weights)
    assert result["status"] == "failed" and result["step"] == 2
    checkpoint = directory / "checkpoints/step-000002"
    restored, manifest, _ = load_checkpoint(
        checkpoint, expected_run_config={"fixture": "current-state-retention"}
    )
    assert restored.step == 2 and manifest["run_state"]["width_guard_retention_failed"]
    assert len(restored.optimizer.state) == len(captured["learner"].optimizer.state) > 0
    assert torch.equal(torch.get_rng_state(), captured["rng"])
    assert all(
        torch.equal(v, restored.network.state_dict()[k])
        for k, v in captured["learner"].network.state_dict().items()
    )
    assert oracle_train.evaluate is measurement
    with pytest.raises(ValueError, match="failed retention"):
        guarded(directory, dataset, weights, resume=checkpoint)
