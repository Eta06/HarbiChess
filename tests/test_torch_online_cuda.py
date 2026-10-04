"""Hardware CUDA gates; skips explicitly on a CPU-only host, never fake CUDA."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_torch_online_checkpoint import equal_tree
from test_torch_online_learner import SOURCE, config, inputs, make_inputs

from harbichess.backends.torch_network import sha256
from harbichess.training.online_objective import OnlineObjectiveConfig, OnlineObjectiveTargets
from harbichess.training.torch_online_checkpoint import (
    CUDA_ONLINE_CHECKPOINT_SCHEMA,
    OnlineCheckpointIntegrityError,
    load_online_checkpoint,
)
from harbichess.training.torch_online_learner import TorchOnlineConfig, TorchOnlineLearner
from harbichess.training.torch_online_objective import online_loss

HAS_CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not HAS_CUDA, reason="actual CUDA GPU unavailable on this host")


def test_device_selection_is_explicit_and_cpu_config_remains_compatible(tmp_path):
    make_inputs(tmp_path)
    learner = TorchOnlineLearner.fresh(config=config(), input_paths=inputs(tmp_path),
                                      source_commit=SOURCE)
    expected = asdict(config())
    expected.pop("device")
    assert learner.run_config["config"] == expected
    for invalid in ("cuda", "cuda:1", "mps", "invalid"):
        with pytest.raises(ValueError):
            replace(config(), device=invalid)
    if not HAS_CUDA:
        with pytest.raises(ValueError, match="unavailable"):
            TorchOnlineLearner.fresh(config=replace(config(), device="cuda:0"),
                                     input_paths=inputs(tmp_path), source_commit=SOURCE)


@cuda
def test_cuda_actual_rollouts_fresh_process_exact_native_resume(tmp_path):
    make_inputs(tmp_path)
    settings = replace(config(), device="cuda:0")
    learner = TorchOnlineLearner.fresh(config=settings, input_paths=inputs(tmp_path),
                                      source_commit=SOURCE)
    frozen_base = {name: tensor.clone() for name, tensor in learner.base.state_dict().items()}
    frozen_material = learner.ema.material_value_linear.weight.clone()
    for _ in range(2):
        learner.train_update()
    manifest = learner.checkpoint(tmp_path / "step2")
    assert manifest["schema"] == CUDA_ONLINE_CHECKPOINT_SCHEMA
    assert manifest["runtime"]["device"] == "cuda:0"
    assert manifest["runtime"]["cuda"]["matmul_tf32"] is False
    assert all(parameter.device.type == "cuda" for parameter in learner.online.parameters())
    for parameter, state in learner.optimizer.state.items():
        assert state["exp_avg"].device == parameter.device
        assert state["exp_avg_sq"].device == parameter.device
    assert all(parameter.grad is None for model in (learner.base, learner.ema)
               for parameter in model.parameters())
    assert all(torch.equal(tensor, frozen_base[name])
               for name, tensor in learner.base.state_dict().items())
    assert torch.equal(learner.ema.material_value_linear.weight, frozen_material)
    assert not torch.backends.cudnn.benchmark
    expected = {"cuda_draw": torch.rand(17, device="cuda:0").cpu().tolist(),
                "records": [learner.train_update() for _ in range(2)]}
    learner.checkpoint(tmp_path / "expected-step4")
    code = """
import json, sys, torch
from pathlib import Path
from dataclasses import replace
from test_torch_online_learner import config, inputs, SOURCE
from harbichess.training.torch_online_learner import TorchOnlineLearner
directory=Path(sys.argv[1])
learner=TorchOnlineLearner.resume(directory/'step2',config=replace(config(),device='cuda:0'),input_paths=inputs(directory),source_commit=SOURCE)
actual={'cuda_draw':torch.rand(17,device='cuda:0').cpu().tolist(),
        'records':[learner.train_update() for _ in range(2)]}
(directory/'restored-records.json').write_text(json.dumps(actual,sort_keys=True))
learner.checkpoint(directory/'restored-step4')
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent),
                                      str(Path(__file__).parents[1] / "src"))),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((tmp_path / "restored-records.json").read_text()) == expected
    for name in ("model.safetensors", "base.safetensors", "ema.safetensors", "actor.json"):
        assert (tmp_path / "expected-step4" / name).read_bytes() == (
            tmp_path / "restored-step4" / name
        ).read_bytes()
    equal_tree(torch.load(tmp_path / "expected-step4/training.pt", weights_only=True,
                          map_location="cpu"),
               torch.load(tmp_path / "restored-step4/training.pt", weights_only=True,
                          map_location="cpu"))
    with pytest.raises(OnlineCheckpointIntegrityError, match="schema/config/source/runtime"):
        load_online_checkpoint(tmp_path / "step2", expected_run_config=learner.run_config,
                               expected_input_paths=inputs(tmp_path),
                               expected_source_commit=SOURCE, device="cpu")


@cuda
def test_cuda_missing_rng_is_rejected_even_with_valid_artifact_checksum(tmp_path):
    make_inputs(tmp_path)
    settings = replace(config(), device="cuda:0")
    learner = TorchOnlineLearner.fresh(config=settings, input_paths=inputs(tmp_path),
                                      source_commit=SOURCE)
    learner.train_update()
    checkpoint = tmp_path / "step1"
    learner.checkpoint(checkpoint)
    state = torch.load(checkpoint / "training.pt", weights_only=True, map_location="cpu")
    state.pop("cuda_rng")
    torch.save(state, checkpoint / "training.pt")
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    manifest["artifacts"]["training.pt"] = sha256(checkpoint / "training.pt")
    (checkpoint / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(OnlineCheckpointIntegrityError, match="CUDA RNG"):
        TorchOnlineLearner.resume(checkpoint, config=settings, input_paths=inputs(tmp_path),
                                  source_commit=SOURCE)


def test_native_device_validation_is_explicit(tmp_path):
    with pytest.raises(OnlineCheckpointIntegrityError, match="unsupported"):
        load_online_checkpoint(tmp_path, expected_run_config={}, expected_input_paths={},
                               expected_source_commit=SOURCE, device="mps")
    assert TorchOnlineConfig.__dataclass_fields__["device"].default == "cpu"


@cuda
def test_cuda_legal_padding_has_exact_zero_gradient():
    policy = torch.tensor([[0.3, -0.4, 2.0]], device="cuda:0", requires_grad=True)
    value = torch.tensor([[0.1, 0.2, 0.3]], device="cuda:0", requires_grad=True)
    targets = OnlineObjectiveTargets(
        legal_masks=np.array([[True, True, False]]), actions=np.array([0]),
        advantages=np.array([0.2]), importance=np.array([1.0]),
        target_wdl=np.array([[0.2, 0.3, 0.5]]),
        base_policy=np.array([[0.6, 0.4, 0.0]]), base_wdl=np.array([[0.3, 0.4, 0.3]]),
    )
    loss = online_loss(policy, value, targets, OnlineObjectiveConfig(0.1, 0.1, 0.1))
    loss.total.backward()
    assert policy.grad[0, 2].item() == 0.0
    assert torch.isfinite(policy.grad).all() and torch.isfinite(value.grad).all()
