import json

import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.training import oracle_train
from harbichess.training.torch_checkpoint import save_checkpoint
from harbichess.training.torch_learner import TorchLearner
from harbichess.training.ufuk_joint_context_guard import guarded


def test_frozen_mutation_stops_before_publication_and_refuses_unsafe_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    weights = tmp_path / "initial.safetensors"
    save_weights(
        weights,
        TorchChessNetwork(
            NetworkConfig(trunk_channels=4, residual_blocks=1), architecture="pairwise"
        ),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}\n")
    directory = tmp_path / "run"
    captured = {}

    def measure(learner, panel):
        return {"policy_ce": 2.0, "value_ce": 0.5, "total_ce": 2.5}

    def producer(directory, dataset, weights, **kwargs):
        directory.mkdir()
        learner = TorchLearner(load_weights(weights))
        measurement = oracle_train.evaluate(learner, None)
        save_checkpoint(
            directory / "checkpoints/step-000000",
            learner=learner,
            sampler=None,
            replay_paths=(dataset / "dataset.json",),
            run_state={"cursor": 0, "evaluations": [{"step": 0, **measurement}]},
            run_config={"fixture": "frozen-state"},
            source_commit="0" * 40,
        )
        with torch.no_grad():
            learner.network.material_value_linear.weight.add_(0.01)
        captured["corrupted"] = learner.network
        oracle_train.evaluate(learner, None)
        pytest.fail("frozen mutation must stop before the next publication")

    monkeypatch.setattr(oracle_train, "evaluate", measure)
    monkeypatch.setattr(oracle_train, "run", producer)
    result = guarded(directory, dataset, weights)
    assert result["status"] == "failed" and result["published_checkpoints_audited"] == 1
    assert sorted(p.name for p in (directory / "checkpoints").iterdir()) == ["step-000000"]
    safe = load_weights(directory / "checkpoints/step-000000/model.safetensors")
    original = load_weights(weights)
    assert all(torch.equal(v, safe.state_dict()[k]) for k, v in original.state_dict().items())
    assert oracle_train.evaluate is measure
    failure = json.loads((directory / "joint-context-frozen-failed-step-000000.json").read_text())
    assert failure["expected"] != failure["actual"]
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    save_weights(unsafe / "model.safetensors", captured["corrupted"])
    with pytest.raises(ValueError, match="frozen checkpoint changed"):
        guarded(directory, dataset, weights, resume=unsafe)
