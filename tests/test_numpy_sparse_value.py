import json
import subprocess
import sys

import chess
import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from harbichess.backends.numpy_sparse_value import NumpySparseValue
from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig


def test_trained_sparse_numpy_matches_torch_mlx_full_metadata_and_special_moves(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    model = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
        value_sparse={"schema": 1, "channels": 16, "hidden": 8},
    )
    boards = [
        chess.Board(),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 99 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 100 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1"),
    ]
    repeated = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeated.push_uci(move)
    boards.append(repeated)
    for origin, move in ((boards[1], "e5d6"), (boards[3], "e1g1"), (boards[4], "a7a8q")):
        changed = origin.copy(stack=True)
        changed.push_uci(move)
        boards.append(changed)
    x = torch.tensor([BoardEncoder().encode_board(b).values for b in boards]).reshape(-1, 8, 8, 104)
    optimizer = torch.optim.AdamW(model.value_sparse_head.parameters(), lr=0.01)
    before = model.value_sparse_head.feature.weight.detach().clone()
    for _ in range(4):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x)[1], torch.arange(len(boards)) % 3)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.value_sparse_head.feature.weight)
    for mlx_layout in (False, True):
        path = tmp_path / f"trained-{mlx_layout}.safetensors"
        save_weights(path, model, mlx_layout=mlx_layout)
        sparse = NumpySparseValue(path)
        native = HarbiChessPairwiseNetwork.from_portable(path)
        actual = native(mx.array(x.numpy()))[1]
        mx.eval(actual)
        with torch.no_grad():
            expected = model(x)[1].numpy()
        np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)
        snapshots = {k: v.copy() for k, v in sparse._weights.items()}
        for row, board in enumerate(boards):
            state = board.fen(), tuple(board.move_stack)
            np.testing.assert_allclose(sparse.logits(board), expected[row], atol=2e-5, rtol=2e-5)
            probability = torch.softmax(torch.tensor(expected[row]), 0).numpy()
            np.testing.assert_allclose(sparse.wdl(board), probability, atol=2e-5, rtol=2e-5)
            assert sparse(board) == pytest.approx(float(probability[0] - probability[2]), abs=2e-5)
            assert (board.fen(), tuple(board.move_stack)) == state
        for key, weight in sparse._weights.items():
            np.testing.assert_array_equal(weight, snapshots[key])
            assert not weight.flags.writeable
        script = (
            "import sys,chess;from pathlib import Path;"
            "from harbichess.backends.numpy_sparse_value import NumpySparseValue;"
            "NumpySparseValue(Path(sys.argv[1]))(chess.Board());"
            "assert 'torch' not in sys.modules and 'mlx.core' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)


def test_numpy_sparse_refuses_unknown_schema_shape_nonfinite_and_dtype(tmp_path):
    shapes = {
        "feature.weight": (4, 832), "feature.bias": (4,), "metadata.weight": (4, 8),
        "hidden.weight": (2, 4), "hidden.bias": (2,), "output.weight": (3, 2),
        "output.bias": (3,),
    }
    weights = {"value_sparse_head." + k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}
    metadata = {"schema": 1, "layout": "torch-oihw", "specification": {
        "architecture": "pairwise", "config": {"input_channels": 104},
        "value_sparse": {"schema": 1, "channels": 4, "hidden": 2},
    }}
    for case in ("schema", "shape", "nonfinite", "dtype", "extra"):
        spec = json.loads(json.dumps(metadata))
        tensors = {k: v.copy() for k, v in weights.items()}
        if case == "schema":
            spec["specification"]["value_sparse"]["schema"] = 2
        elif case == "shape":
            tensors["value_sparse_head.output.bias"] = np.zeros(4, dtype=np.float32)
        elif case == "nonfinite":
            tensors["value_sparse_head.output.bias"][0] = np.nan
        elif case == "dtype":
            tensors["value_sparse_head.output.bias"] = np.zeros(3, dtype=np.float64)
        else:
            tensors["value_sparse_head.unexpected"] = np.zeros(1, dtype=np.float32)
        path = tmp_path / f"{case}.safetensors"
        save_file(tensors, path, metadata={"harbichess": json.dumps(spec)})
        with pytest.raises(ValueError):
            NumpySparseValue(path)
