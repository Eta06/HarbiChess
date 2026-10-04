import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.core.network_config import NetworkConfig
from harbichess.training.online_objective import OnlineObjectiveConfig, OnlineObjectiveTargets
from harbichess.training.torch_online_checkpoint import (
    OnlineCheckpoint,
    OnlineCheckpointIntegrityError,
    load_online_checkpoint,
    save_online_checkpoint,
)
from harbichess.training.torch_online_objective import online_loss

SOURCE = "a" * 40
CONFIG = {"schema": "synthetic-resume-audit-not-self-play", "ema_decay": 0.9}


def fixture():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(271)
    np.random.seed(271)
    random.seed(271)
    model = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=4, residual_blocks=1, policy_channels=1, value_channels=1, value_hidden=4
        ),
        architecture="pairwise",
        invariant={"channels": 2, "blocks": 1, "hidden": 4},
    )
    model.material_value_linear.requires_grad_(False)
    base, ema = copy.deepcopy(model).eval(), copy.deepcopy(model).eval()
    base.requires_grad_(False)
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    return OnlineCheckpoint(
        model,
        base,
        ema,
        optimizer,
        random.Random(721),
        {
            "update": 0,
            "audit_actions": [],
            "games": [{"root_fen": chess.STARTING_FEN, "moves": []}],
        },
        {},
    )


def advance(bundle):
    """Synthetic real gradient update; not a legal game or learning experiment."""
    inputs = torch.rand(2, 8, 8, 104)
    actions = torch.tensor([legal_action_indices(chess.Board())] * 2)
    policy, value = bundle.online.masked_policy_value(inputs, actions)
    with torch.no_grad():
        bp, bv = bundle.base.masked_policy_value(inputs, actions)
        _, ev = bundle.ema.masked_policy_value(inputs, actions)
    chosen = np.array([bundle.actor_rng.randrange(actions.shape[1]) for _ in range(2)])
    data = OnlineObjectiveTargets(
        legal_masks=np.ones(policy.shape, dtype=np.bool_),
        actions=chosen,
        advantages=np.random.uniform(-1, 1, 2) * random.random(),
        importance=np.ones(2),
        target_wdl=ev.softmax(1).numpy()[:, [2, 1, 0]],
        base_policy=bp.softmax(1).numpy(),
        base_wdl=bv.softmax(1).numpy(),
    )
    bundle.optimizer.zero_grad(set_to_none=True)
    online_loss(policy, value, data, OnlineObjectiveConfig(0.03, 0.2, 0.02)).total.backward()
    torch.nn.utils.clip_grad_norm_(bundle.online.parameters(), 5, error_if_nonfinite=True)
    bundle.optimizer.step()
    with torch.no_grad():
        for target, source in zip(bundle.ema.parameters(), bundle.online.parameters(), strict=True):
            target.mul_(0.9).add_(source, alpha=0.1)
    bundle.run_state["update"] += 1
    bundle.run_state["audit_actions"].append(chosen.tolist())


def save(path, bundle, book):
    return save_online_checkpoint(
        path,
        online=bundle.online,
        base=bundle.base,
        ema=bundle.ema,
        optimizer=bundle.optimizer,
        actor_rng=bundle.actor_rng,
        run_state=bundle.run_state,
        run_config=CONFIG,
        input_paths={"book": book},
        source_commit=SOURCE,
        update=bundle.run_state["update"],
    )


def load(path, book, **changes):
    arguments = dict(
        expected_run_config=CONFIG,
        expected_input_paths={"book": book},
        expected_source_commit=SOURCE,
    )
    return load_online_checkpoint(path, **(arguments | changes))


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            equal_tree(a, e)
    else:
        assert actual == expected


def test_fresh_process_all_three_networks_adam_rng_cursor_and_next_update_exact(tmp_path):
    book = tmp_path / "book.json"
    book.write_text('{"immutable":true}\n')
    bundle = fixture()
    for _ in range(2):
        advance(bundle)
    saved = save(tmp_path / "step2", bundle, book)
    advance(bundle)
    expected = save(tmp_path / "expected-step3", bundle, book)
    code = """
import sys
from pathlib import Path
import torch
from test_torch_online_checkpoint import load, save, advance
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
directory=Path(sys.argv[1]); book=directory/'book.json'
bundle=load(directory/'step2',book)
advance(bundle)
save(directory/'restored-step3',bundle,book)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(Path(__file__).parent), str(Path(__file__).parents[1] / "src"))
        ),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual_path = tmp_path / "restored-step3"
    actual = json.loads((actual_path / "checkpoint.json").read_text())
    assert saved["transfer"] == "full-online-training"
    assert actual["update"] == expected["update"] == 3
    assert (actual_path / "actor.json").read_bytes() == (
        tmp_path / "expected-step3/actor.json"
    ).read_bytes()
    for name in ("model.safetensors", "base.safetensors", "ema.safetensors"):
        assert actual["artifacts"][name] == expected["artifacts"][name]
    equal_tree(
        torch.load(actual_path / "training.pt", weights_only=True),
        torch.load(tmp_path / "expected-step3/training.pt", weights_only=True),
    )


@pytest.mark.parametrize("what", ["book", "actor.json", "training.pt", "ema.safetensors"])
def test_reject_corrupted_inputs_and_every_state_component(tmp_path, what):
    book = tmp_path / "book.json"
    book.write_text("original\n")
    save(tmp_path / "checkpoint", fixture(), book)
    path = book if what == "book" else tmp_path / "checkpoint" / what
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(OnlineCheckpointIntegrityError, match="mismatch"):
        load(tmp_path / "checkpoint", book)


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_run_config": CONFIG | {"ema_decay": 0.8}},
        {"expected_source_commit": "b" * 40},
        {"expected_input_paths": {}},
    ],
)
def test_reject_changed_algorithm_source_and_immutable_input_names(tmp_path, changes):
    book = tmp_path / "book.json"
    book.write_text("original\n")
    save(tmp_path / "checkpoint", fixture(), book)
    with pytest.raises(OnlineCheckpointIntegrityError):
        load(tmp_path / "checkpoint", book, **changes)


def test_never_overwrite_published_checkpoint_or_allow_weights_only_resume(tmp_path):
    book = tmp_path / "book.json"
    book.write_text("original\n")
    bundle = fixture()
    save(tmp_path / "checkpoint", bundle, book)
    before = sha256(tmp_path / "checkpoint/checkpoint.json")
    with pytest.raises(FileExistsError):
        save(tmp_path / "checkpoint", bundle, book)
    assert sha256(tmp_path / "checkpoint/checkpoint.json") == before
    (tmp_path / "weights-only").mkdir()
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "weights-only", book)


def test_reject_parameter_storage_alias_and_nonfinite_adam_state(tmp_path):
    book = tmp_path / "book.json"
    book.write_text("original\n")
    bundle = fixture()
    bundle.ema = copy.copy(bundle.base)
    with pytest.raises(ValueError, match="storage must not alias"):
        save(tmp_path / "alias", bundle, book)
    assert not (tmp_path / "alias").exists()
    bundle = fixture()
    advance(bundle)
    next(iter(bundle.optimizer.state.values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite tensors"):
        save(tmp_path / "nan", bundle, book)
    assert not (tmp_path / "nan").exists()
