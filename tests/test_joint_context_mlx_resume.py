import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from test_replay_schema import scripted_game

from harbichess.backends.invariant_value_network import InvariantValueConfig
from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.core.network_config import NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from harbichess.training.config import LearnerConfig
from harbichess.training.learner import MLXLearner
from harbichess.training.resume import ResumeState


def make_learner():
    network = HarbiChessPairwiseNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        invariant_config=InvariantValueConfig(4, 1, 4),
        policy_context={"schema": 1, "blocks": 2, "heads": 2},
    )
    network.material_value_linear.freeze()
    return MLXLearner(network, config=LearnerConfig(learning_rate=1e-3))


def test_joint_context_full_native_mlx_optimizer_sampler_and_next_update_resume(tmp_path):
    mx.set_default_device(mx.cpu)
    _, game = scripted_game()
    records = records_from_game(game, run_id="joint-context-resume")
    sampler = GameBalancedSampler(records, seed=20261024)
    learner = make_learner()
    for _ in range(2):
        learner.train_step(build_training_batch(sampler.sample(2)))
    state = ResumeState(
        schema_version=1,
        run_id="joint-context-resume",
        checkpoint_id="step-2",
        source_commit="a" * 40,
        created_at="2026-10-04T02:00:00Z",
        training_step=2,
        lifetime_games=1,
        generation_games=1,
        training_elapsed_seconds=0,
        replay_samples=4,
        replay_cursor=4,
        model_file="model.safetensors",
        optimizer_file="optimizer.safetensors",
        rng_file="sampler-rng.json",
    )
    path = tmp_path / "checkpoint"
    saved = save_training_checkpoint(path, state=state, learner=learner, sampler=sampler)
    next_records = sampler.sample(3)
    expected_metrics = learner.train_step(build_training_batch(next_records))
    restored, restored_sampler = make_learner(), GameBalancedSampler(records, seed=999)
    loaded = load_training_checkpoint(path, learner=restored, sampler=restored_sampler)
    assert loaded == saved and restored.step == 2
    assert restored_sampler.sample(3) == next_records
    actual_metrics = restored.train_step(build_training_batch(next_records))
    assert actual_metrics == expected_metrics and restored.step == learner.step == 3
    for expected_tree, actual_tree in (
        (learner.network.parameters(), restored.network.parameters()),
        (learner.optimizer.state, restored.optimizer.state),
    ):
        mx.eval(expected_tree, actual_tree)
        expected, actual = dict(tree_flatten(expected_tree)), dict(tree_flatten(actual_tree))
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(np.array(expected[name]), np.array(actual[name]))
