import chess
import torch

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch


def test_joint_context_real_shared_value_policy_updates_and_exact_full_torch_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
        policy_context={"schema": 1, "blocks": 2, "heads": 2},
    )
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(not name.startswith("material_value_linear."))
    before = {k: v.detach().clone() for k, v in network.state_dict().items()}
    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci("e2e4")
    inputs = torch.tensor([BoardEncoder().encode_board(b).values for b in boards])
    policies, masks = torch.zeros(2, 4672), torch.zeros(2, 4672, dtype=torch.bool)
    for index, board in enumerate(boards):
        legal = legal_action_indices(board)
        masks[index, list(legal)] = True
        policies[index, legal[0]], policies[index, legal[1]] = 0.8, 0.2
    batch = TorchTrainingBatch(
        inputs.reshape(2, 8, 8, 104), policies, masks,
        torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]]), torch.ones(2),
    )
    learner = TorchLearner(network, config=LearnerConfig(learning_rate=0.01))
    for _ in range(3):
        learner.train_step(batch)
    states = network.state_dict()
    for prefix in ("stem.", "value_", "pair_", "policy_context_blocks."):
        assert any(not torch.equal(value, before[k]) for k, value in states.items()
                   if k.startswith(prefix))
    assert all(torch.equal(value, before[k]) for k, value in states.items()
               if k.startswith("material_value_linear."))
    marker = tmp_path / "data.json"
    marker.write_text('{"fixture":"joint-soft-WDL"}\n')
    path, config = tmp_path / "checkpoint", {"scope": "joint-full-native"}
    save_checkpoint(
        path, learner=learner, sampler=None, replay_paths=(marker,),
        run_state={"cursor": 3}, run_config=config, source_commit="a" * 40,
    )
    indices = tuple(torch.randint(2, (4,)).tolist())
    expected = learner.train_step(batch.select(indices))
    restored, manifest, _ = load_checkpoint(path, expected_run_config=config)
    assert manifest["step"] == restored.step == 3
    assert tuple(torch.randint(2, (4,)).tolist()) == indices
    assert restored.train_step(batch.select(indices)) == expected
    for name, value in learner.network.state_dict().items():
        assert torch.equal(value, restored.network.state_dict()[name])
    original, actual = learner.optimizer.state_dict(), restored.optimizer.state_dict()
    assert original["param_groups"] == actual["param_groups"]
    assert original["state"].keys() == actual["state"].keys()
    for identifier, entry in original["state"].items():
        assert all(torch.equal(value, actual["state"][identifier][name])
                   for name, value in entry.items())
