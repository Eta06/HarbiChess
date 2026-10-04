import gzip
import json
import subprocess
import sys

import chess
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
import torch

from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.chess.actions import move_to_action
from harbichess.core.network_config import NetworkConfig
from harbichess.training import oracle_train
from harbichess.training.oracle_data import load_panels
from harbichess.training.prepared_oracle import PreparedOracle, digest, prepare
from harbichess.training.prepared_oracle_torch import load_prepared_panels
from harbichess.training.torch_checkpoint import load_checkpoint
from harbichess.training.torch_learner import TorchLearner


def make_dataset(path):
    path.mkdir()
    files = {}
    for split, family, histories in (
        ("train", 0, (("e2e4",), ("d2d4",), ("c2c4",))),
        ("validation", 1, (("e2e4",), ("e2e4", "e7e5"), ("d2d4", "d7d5"))),
    ):
        rows = []
        for moves in histories:
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            rows.append(
                {
                    "root_fen": chess.STARTING_FEN,
                    "moves": list(moves),
                    "fen": board.fen(),
                    "position_key": " ".join(board.fen().split()[:4]),
                    "split": split,
                    "family": family,
                    "legal": [[m.uci(), move_to_action(board, m)] for m in legal],
                    "policy": [
                        [m.uci(), move_to_action(board, m), p]
                        for m, p in zip(legal[:2], (0.8, 0.2), strict=True)
                    ],
                    "wdl": [0.2, 0.5, 0.3],
                }
            )
        filename = split + ".json.gz"
        data = {
            "schema": 1,
            "target_semantics": "engine-reference",
            "job": {
                "split": split,
                "family": family,
            },
            "rows": rows,
        }
        (path / filename).write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
        files[filename] = digest(path / filename)
    (path / "dataset.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "completed",
                "rows": 6,
                "files": files,
            }
        )
        + "\n"
    )
    (path / "metadata.json").write_text('{"fixture":"prepared-native"}\n')


def test_prepared_native_caps_overlap_every_field_and_torch_free_reader(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    make_dataset(source)
    receipt = prepare(cache, source=source)
    assert receipt["every_row_full_input_exact"] and receipt["rows"] == 6
    before = np.random.get_state()
    original = load_panels(source, max_train_rows=2, max_validation_rows=1, seed=31)
    cached = load_prepared_panels(cache, source, max_train_rows=2, max_validation_rows=1, seed=31)
    assert original[2] == cached[2] and original[2]["removed_validation_position_overlap"] == 1
    for plain, packed in zip(original[:2], cached[:2], strict=True):
        indices = (0, 0, plain.size - 1)
        a, b = plain.select(indices), packed.select(indices)
        for name in ("inputs", "policies", "legal_masks", "wdl", "value_weights"):
            assert torch.equal(getattr(a, name), getattr(b, name))
    assert np.array_equal(np.random.get_state()[1], before[1])
    script = (
        "import sys;from pathlib import Path;"
        "from harbichess.training.prepared_oracle import PreparedOracle;"
        "p=PreparedOracle(Path(sys.argv[1]),source=Path(sys.argv[2])).panels()[0];"
        "p.select((0,));assert 'torch' not in sys.modules and 'mlx.core' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script, str(cache), str(source)], check=True)
    with pytest.raises(FileExistsError):
        prepare(cache, source=source)


def test_prepared_checksum_source_and_unknown_schema_refusal(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    make_dataset(source)
    prepare(cache, source=source)
    path = cache / "prepared.json"
    original = path.read_text()
    changed = json.loads(original)
    changed["schema"] = 2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="schema"):
        PreparedOracle(cache, source=source)
    path.write_text(original)
    metadata = source / "metadata.json"
    content = metadata.read_bytes()
    metadata.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="source metadata changed"):
        PreparedOracle(cache, source=source)
    metadata.write_bytes(content)
    array = cache / "wdl.npy"
    content = array.read_bytes()
    array.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        PreparedOracle(cache, source=source)


def test_prepared_real_training_exact_cached_resume_and_profile_refusal(tmp_path):
    torch.set_num_threads(1)
    source, cache = tmp_path / "source", tmp_path / "cache"
    make_dataset(source)
    prepare(cache, source=source)
    weights = tmp_path / "weights.safetensors"
    torch.manual_seed(3132)
    save_weights(
        weights,
        TorchChessNetwork(
            NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
            architecture="pairwise",
            invariant={"channels": 4, "blocks": 1, "hidden": 4},
        ),
    )
    kwargs = dict(max_steps=2, interval=1, patience=10, batch_size=2, seed=31, prepared_cache=cache)
    whole, split = tmp_path / "whole", tmp_path / "split"
    oracle_train.run(whole, source, weights, **kwargs)
    oracle_train.run(split, source, weights, stop_at=1, **kwargs)
    with pytest.raises(ValueError, match="config"):
        oracle_train.run(
            split,
            source,
            weights,
            resume=split / "checkpoints/step-000001",
            **{**kwargs, "prepared_cache": None},
        )
    oracle_train.run(split, source, weights, resume=split / "checkpoints/step-000001", **kwargs)

    def loaded(path):
        expected_config = json.loads((path / "checkpoint.json").read_text())["run_config"]
        return load_checkpoint(path, expected_run_config=expected_config)

    original, _, _ = loaded(whole / "checkpoints/step-000002")
    restored, manifest, _ = loaded(split / "checkpoints/step-000002")
    assert manifest["run_config"]["prepared_oracle"]["schema"] == 1
    for name, value in original.network.state_dict().items():
        assert torch.equal(value, restored.network.state_dict()[name])
    expected, actual = original.optimizer.state_dict(), restored.optimizer.state_dict()
    assert expected["param_groups"] == actual["param_groups"]
    for identifier, entry in expected["state"].items():
        assert all(
            torch.equal(value, actual["state"][identifier][name]) for name, value in entry.items()
        )
    uncached = tmp_path / "uncached"
    oracle_train.run(uncached, source, weights, **{**kwargs, "prepared_cache": None})
    reference, _, _ = loaded(uncached / "checkpoints/step-000002")
    for name, value in original.network.state_dict().items():
        assert torch.equal(value, reference.network.state_dict()[name])


def test_prepared_native_soft_targets_real_mlx_loss_gradient_and_update(tmp_path):
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(1)
    source, cache = tmp_path / "source", tmp_path / "cache"
    make_dataset(source)
    prepare(cache, source=source)
    panel = PreparedOracle(cache, source=source).panels()[0]
    data = panel.select((0, 1))
    torch.manual_seed(3133)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    path = tmp_path / "weights.safetensors"
    save_weights(path, network)
    model = HarbiChessPairwiseNetwork.from_portable(path)
    model.material_value_linear.freeze()
    x, targets, legal, wdl = (
        mx.array(getattr(data, name)) for name in ("inputs", "policies", "legal_masks", "wdl")
    )

    def loss(model):
        policy, value = model(x)
        masked = mx.where(legal, policy, -1e9)
        pol = -(targets * (masked - mx.logsumexp(masked, axis=1, keepdims=True))).sum(1).mean()
        val = -(wdl * (value - mx.logsumexp(value, axis=1, keepdims=True))).sum(1).mean()
        return pol + val

    value, gradients = nn.value_and_grad(model, loss)(model)
    mx.eval(value, gradients)
    torch_batch = load_prepared_panels(cache, source)[0].select((0, 1))
    expected = TorchLearner(network)._loss(torch_batch)[0]
    assert float(value) == pytest.approx(float(expected.detach()), abs=2e-5)
    assert float(mx.abs(gradients["stem"]["weight"]).sum()) > 0
    before = np.array(model.stem.weight)
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    assert not np.array_equal(before, np.array(model.stem.weight))


def test_prepared_cli_uses_cache_in_actual_training_checkpoint(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    make_dataset(source)
    prepare(cache, source=source)
    weights = tmp_path / "weights.safetensors"
    save_weights(
        weights,
        TorchChessNetwork(
            NetworkConfig(trunk_channels=2, residual_blocks=1, value_hidden=2),
            architecture="pairwise",
            invariant={"channels": 2, "blocks": 1, "hidden": 2},
        ),
    )
    directory = tmp_path / "cli"
    subprocess.run(
        [
            sys.executable, "-m", "harbichess.training.oracle_train", str(directory),
            "--dataset", str(source), "--weights", str(weights),
            "--prepared-cache", str(cache), "--max-steps", "1", "--interval", "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((directory / "checkpoints/step-000001/checkpoint.json").read_text())
    assert manifest["run_config"]["prepared_oracle"]["manifest_sha256"] == digest(
        cache / "prepared.json"
    )
    assert sum(name.endswith(".npy") for name in manifest["replay"]) == 12
    assert manifest["step"] == 1 and manifest["transfer"] == "full-training"
