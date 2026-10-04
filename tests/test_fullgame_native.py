import copy
import gzip
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.selfplay.online_epoch import deserialize_policy_epoch
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.torch_fullgame_learner import (
    TorchFullGameConfig,
    TorchFullGameLearner,
    canonical,
)
from harbichess.training.torch_fullgame_ppo import FullGamePPOConfig, FullGamePPOTrainConfig

SOURCE = "b" * 40


def config(device="cpu"):
    return TorchFullGameConfig(
        seed=91,
        actors=OnlineActorConfig(8, 4, False, 1.0),
        objective=FullGamePPOConfig(0.1, 0.01, 0.03, 0.2, 0.02),
        schedule=FullGamePPOTrainConfig(32, 2, 5.0),
        epoch_steps=4,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device=device,
    )


def inputs(root):
    return {"initial_weights": root / "initial.safetensors", "book": root / "book.json"}


def make_inputs(root):
    torch.manual_seed(123)
    model = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=4, residual_blocks=1, policy_channels=2, value_channels=2, value_hidden=4
        ),
        architecture="pairwise",
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("material_value_linear."):
                p.copy_(torch.linspace(-0.29137, 0.41319, p.numel()).reshape(p.shape))
    save_weights(root / "initial.safetensors", model)
    rows = [
        {"source_game": name, "root_ply": 0, "opening": {"root_fen": fen, "moves": []}}
        for name, fen in [
            ("white", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
            ("black", "8/8/8/8/8/6k1/5q2/7K b - - 0 1"),
        ]
    ]
    (root / "book.json").write_text(json.dumps({"schema": 1, "splits": {"train": rows}}))


def fresh(root, device="cpu"):
    return TorchFullGameLearner.fresh(
        config=config(device), input_paths=inputs(root), source_commit=SOURCE
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, tuple | list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def test_actual_closed_epoch_terminal_caps_base_and_nonzero_material(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    before = copy.deepcopy(learner.online.state_dict())
    base = copy.deepcopy(learner.base.state_dict())
    records = [learner.train_epoch(), learner.train_epoch()]
    assert learner.epoch == 2 and learner.actors.steps == 8
    assert learner.optimizer_accepted_updates > 0
    assert learner.optimizer_accepted_updates != learner.actors.steps
    assert any(
        not torch.equal(value, learner.online.state_dict()[name]) for name, value in before.items()
    )
    terminal_sides = set()
    for record in records:
        epoch = deserialize_policy_epoch(canonical(record["collection"]))
        targets = build_fullgame_targets(learner.actors.rules, epoch, claim_draw=False)
        assert targets.targets
        assert record["target_counts"]["excluded_actions"] > 0
        for row in targets.targets:
            outcome = learner.actors.rules.outcome(row.transition.post, claim_draw=False)
            # Earlier rows inherit final episode outcome, never their nonterminal poststate.
            if outcome:
                side = learner.actors.rules.view(row.transition.pre).side_to_move
                terminal_sides.add(side)
                expected = outcome.value_for(side)
                assert row.target_wdl == (
                    float(expected == 1),
                    float(expected == 0),
                    float(expected == -1),
                )
        assert not any(
            p.grad is not None for m in (learner.base, learner.behavior) for p in m.parameters()
        )
    assert len(terminal_sides) == 2
    equal_tree(base, learner.base.state_dict())
    for name, original in before.items():
        if name.startswith("material_value_linear."):
            for model in (learner.online, learner.base, learner.behavior):
                equal_tree(original, model.state_dict()[name])
    learner.checkpoint(tmp_path / "native2")
    restored = TorchFullGameLearner.resume(
        tmp_path / "native2", config=config(), input_paths=inputs(tmp_path), source_commit=SOURCE
    )
    equal_tree(learner.optimizer.state_dict(), restored.optimizer.state_dict())
    assert learner.actors.cursor() == restored.actors.cursor()
    assert learner.last_epoch_gzip == restored.last_epoch_gzip


def test_checkpoint_rejects_partial_epoch_and_unknown_caps_have_zero_updates(tmp_path):
    make_inputs(tmp_path)
    cfg = replace(config(), actors=OnlineActorConfig(8, 1, False, 1.0), epoch_steps=1)
    # Starting position cannot end after one fresh move.
    book = {
        "schema": 1,
        "splits": {"train": [{"source_game": "initial", "root_ply": 0, "opening": {"moves": []}}]},
    }
    (tmp_path / "book.json").write_text(json.dumps(book))
    learner = TorchFullGameLearner.fresh(
        config=cfg, input_paths=inputs(tmp_path), source_commit=SOURCE
    )
    record = learner.train_epoch()
    assert record["target_counts"]["normal_cap_games"] == 8
    assert record["training"]["trained_transitions"] == 0
    assert learner.optimizer_accepted_updates == learner.optimizer_attempted_updates == 0
    learner.checkpoint(tmp_path / "cap-native")
    learner.closed = False
    with pytest.raises(ValueError, match="partial"):
        learner.checkpoint(tmp_path / "partial")
    assert not (tmp_path / "partial").exists()


def test_native_integrity_source_config_and_budget_paths_are_strict(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    learner.train_epoch()
    learner.checkpoint(tmp_path / "native")
    with pytest.raises(ValueError, match="mismatch"):
        TorchFullGameLearner.resume(
            tmp_path / "native",
            config=config(),
            input_paths=inputs(tmp_path),
            source_commit="c" * 40,
        )
    with pytest.raises(ValueError, match="mismatch"):
        TorchFullGameLearner.resume(
            tmp_path / "native",
            config=replace(config(), learning_rate=2e-4),
            input_paths=inputs(tmp_path),
            source_commit=SOURCE,
        )
    path = tmp_path / "native/last-frozen-epoch.json.gz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        TorchFullGameLearner.resume(
            tmp_path / "native", config=config(), input_paths=inputs(tmp_path), source_commit=SOURCE
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="actualCUDA host required"
            ),
        ),
    ],
)
def test_new_process_two_epochs_vs_one_then_one_native_bytes(tmp_path, device):
    make_inputs(tmp_path)
    cfg = config(device)
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg)))
    for name in ("whole", "split"):
        (tmp_path / name).mkdir()
    worker = """
import json,sys
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOConfig,FullGamePPOTrainConfig
from harbichess.training.torch_fullgame_learner import TorchFullGameConfig,TorchFullGameLearner
root=Path(sys.argv[1]);out=root/sys.argv[2];resume=sys.argv[3]=='resume';steps=int(sys.argv[4])
c=json.loads((root/'config.json').read_text());c['actors']=OnlineActorConfig(**c['actors']);c['objective']=FullGamePPOConfig(**c['objective']);c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
kw=dict(config=TorchFullGameConfig(**c),input_paths={'initial_weights':root/'initial.safetensors','book':root/'book.json'},source_commit='b'*40)
l=TorchFullGameLearner.resume(out/'epoch1',**kw) if resume else TorchFullGameLearner.fresh(**kw)
for _ in range(steps):
 l.train_epoch();(out/f'epoch-{l.epoch}.json.gz').write_bytes(l.last_epoch_gzip)
 l.checkpoint(out/f'epoch{l.epoch}')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    for args in [("whole", "fresh", "2"), ("split", "fresh", "1"), ("split", "resume", "1")]:
        subprocess.run(
            [sys.executable, "-c", worker, str(tmp_path), *args],
            check=True,
            env=env,
            capture_output=True,
            timeout=120,
        )
    for i in (1, 2):
        assert (tmp_path / "whole" / f"epoch-{i}.json.gz").read_bytes() == (
            tmp_path / "split" / f"epoch-{i}.json.gz"
        ).read_bytes()
    for name in (
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    ):
        assert (tmp_path / "whole/epoch2" / name).read_bytes() == (
            tmp_path / "split/epoch2" / name
        ).read_bytes(), name
    a = torch.load(tmp_path / "whole/epoch2/training.pt", weights_only=True)
    b = torch.load(tmp_path / "split/epoch2/training.pt", weights_only=True)
    equal_tree(a, b)
    assert (
        json.loads(gzip.decompress((tmp_path / "whole/epoch-2.json.gz").read_bytes()))[
            "optimizer_accepted_updates"
        ]
        > 0
    )


def test_runner_expired_budget_and_dirty_source_refuse_before_mutation(tmp_path, monkeypatch):
    import time

    from harbichess.training.torch_fullgame_run import run_fullgame

    make_inputs(tmp_path)
    kw = dict(
        config=config(),
        input_paths=inputs(tmp_path),
        source_commit=SOURCE,
        max_epochs=2,
        checkpoint_interval=1,
        deadline_epoch=time.time() - 1,
        memory_max_bytes=2**63,
        disk_min_free_bytes=1,
    )
    with pytest.raises(ValueError, match="whole-run deadline"):
        run_fullgame(tmp_path / "expired", **kw)
    assert not (tmp_path / "expired").exists()
    kw["deadline_epoch"] = time.time() + 60
    with pytest.raises(ValueError, match="clean source"):
        run_fullgame(tmp_path / "dirty", **kw)
    assert not (tmp_path / "dirty").exists()


def test_deadline_abort_in_collection_refuses_partial_native(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("hard-deadline")

    with pytest.raises(TimeoutError, match="hard-deadline"):
        learner.train_epoch(guard=guard)
    assert not learner.closed and learner.actors.steps == 1 and learner.epoch == 0
    with pytest.raises(ValueError, match="partial"):
        learner.checkpoint(tmp_path / "partial")


def test_whole_runner_freshprocess_control_and_exact_resume_budget(tmp_path, monkeypatch):
    import time

    import harbichess.training.torch_fullgame_run as runner

    make_inputs(tmp_path)
    # Clean source validation is tested separately and NEVER bypassable in CLI.
    # This unit seam exercises budget metadata and publication without committing scratch.
    monkeypatch.setattr(runner, "clean_source", lambda source: Path(__file__).resolve().parents[1])
    kw = dict(
        config=config(),
        input_paths=inputs(tmp_path),
        source_commit=SOURCE,
        max_epochs=2,
        checkpoint_interval=1,
        deadline_epoch=time.time() + 120,
        memory_max_bytes=2**63,
        disk_min_free_bytes=1,
    )
    directory = tmp_path / "run"
    result = runner.run_fullgame(directory, stop_at=1, **kw)
    assert result["epoch"] == 1 and result["status"] == "completed"
    resume = directory / "checkpoints/epoch-00000001"
    with pytest.raises(ValueError, match="deadline changed"):
        runner.run_fullgame(
            directory, resume=resume, **(kw | {"deadline_epoch": kw["deadline_epoch"] + 1})
        )
    result = runner.run_fullgame(directory, resume=resume, **kw)
    assert result["epoch"] == 2 and result["closed_boundary"]
    runner.verify_epochs(directory, 2, result["sample_chain_sha256"])
    assert not (directory / ".online-active.json").exists()


def test_frozen_material_signedzero_byte_change_rejected_at_save_and_load(tmp_path):
    from harbichess.backends.torch_network import load_weights
    from harbichess.training.torch_fullgame_learner import tensor_bits_equal

    make_inputs(tmp_path)
    model = load_weights(tmp_path / "initial.safetensors")
    name, p = next(
        (n, p) for n, p in model.named_parameters() if n.startswith("material_value_linear.")
    )
    with torch.no_grad():
        p.reshape(-1)[0] = 0.0
    (tmp_path / "initial.safetensors").rename(tmp_path / "nonzero-initial.safetensors")
    save_weights(tmp_path / "initial.safetensors", model)
    learner = fresh(tmp_path)
    learner.checkpoint(tmp_path / "valid")
    # Numeric equality would accept this frozen parameter corruption.
    target = dict(learner.online.named_parameters())[name]
    with torch.no_grad():
        target.reshape(-1)[0] = -0.0
    assert torch.equal(target, learner.base.state_dict()[name])
    assert not tensor_bits_equal(target, learner.base.state_dict()[name])
    assert not tensor_bits_equal(torch.tensor(0.0), torch.tensor(-0.0))
    assert tensor_bits_equal(torch.tensor(3.0), torch.tensor(3.0))
    with pytest.raises(ValueError, match="inherited material"):
        learner.checkpoint(tmp_path / "bad-save")
    assert not (tmp_path / "bad-save").exists()

    # Build a coherently hash-updated malformed container: strict loader must
    # reject frozen provenance rather than relying only on file checksums.
    from harbichess.backends.torch_network import sha256

    native = tmp_path / "valid"
    training = torch.load(native / "training.pt", weights_only=True)
    training["online"][name].reshape(-1)[0] = -0.0
    torch.save(training, native / "training.pt")
    portable = load_weights(native / "model.safetensors")
    with torch.no_grad():
        dict(portable.named_parameters())[name].reshape(-1)[0] = -0.0
    (native / "model.safetensors").rename(native / "original-model.safetensors")
    save_weights(native / "model.safetensors", portable)
    manifest = json.loads((native / "checkpoint.json").read_text())
    for file in ("training.pt", "model.safetensors"):
        manifest["artifacts"][file] = sha256(native / file)
    (native / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inherited material"):
        TorchFullGameLearner.resume(
            native, config=config(), input_paths=inputs(tmp_path), source_commit=SOURCE
        )
