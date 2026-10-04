import chess
import mlx.core as mx
import numpy as np
import pytest
import torch

from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.packed_encoding import decode_positions, pack_board
from harbichess.core.network_config import NetworkConfig


def test_packed_exact_all104_histories_special_moves_and_real_torch_mlx(tmp_path):
    torch.set_num_threads(1)
    boards = []
    for fen, moves in (
        (chess.STARTING_FEN, ("g1f3", "g8f6", "f3g1", "f6g8") * 2),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 99 1", ("e5d6",)),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 100 1", ("e4d3",)),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ("e1g1", "e8c8")),
        ("8/P3k3/8/8/8/8/8/4K3 w - - 0 1", ("a7a8n",)),
    ):
        board = chess.Board(fen)
        boards.append(board.copy(stack=True))
        for move in moves:
            board.push_uci(move)
            boards.append(board.copy(stack=True))
    packed = [pack_board(board) for board in boards]
    pieces, metadata, ep = (
        np.array([p[i] for p in packed], dtype=dtype)
        for i, dtype in enumerate((np.uint64, np.float32, np.uint8))
    )
    decoded = decode_positions(pieces, metadata, ep)
    expected = np.array([BoardEncoder().encode_board(b).values for b in boards], dtype=np.float32)
    np.testing.assert_array_equal(decoded.reshape(len(boards), -1), expected)
    # Explicit big-endian storage still decodes the same square bits.
    np.testing.assert_array_equal(decode_positions(pieces.astype(">u8"), metadata, ep), decoded)
    torch.manual_seed(1030)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    path = tmp_path / "weights.safetensors"
    save_weights(path, network)
    mlx = HarbiChessPairwiseNetwork.from_portable(path)
    actual = mlx(mx.array(decoded))
    mx.eval(actual)
    with torch.no_grad():
        original = network(torch.from_numpy(expected.reshape(-1, 8, 8, 104)))
        cache = network(torch.from_numpy(decoded))
    for a, b, c in zip(original, cache, actual, strict=True):
        assert torch.equal(a, b)
        np.testing.assert_allclose(a.numpy(), np.array(c), atol=2e-5, rtol=2e-5)


def test_packed_refuses_corrupt_metadata_ep_shape_and_dtype():
    p, m, e = pack_board(chess.Board())
    for case in ("nan", "ep", "shape", "dtype", "double-ep"):
        pieces, metadata, ep = p[None].copy(), m[None].copy(), np.array([e], dtype=np.uint8)
        if case == "nan":
            metadata[0, 6] = np.nan
        elif case == "ep":
            ep[0] = 65
        elif case == "shape":
            pieces = pieces[:, :7]
        elif case == "dtype":
            metadata = metadata.astype(np.float64)
        else:
            metadata[0, 5] = 1
        with pytest.raises(ValueError):
            decode_positions(pieces, metadata, ep)
