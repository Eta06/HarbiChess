"""Sparse fast adapter is identical to production inference, no speed claim."""

import os
from pathlib import Path

import chess
import numpy as np
import torch
from value import NeuralValue

from harbichess.backends.torch_network import load_weights
from harbichess.chess.actions import legal_action_indices
from harbichess.search.evaluator import _softmax
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def test_cpu_value_adapter_actual_e8_and_positional_checkpoint():
    torch.set_num_threads(1)
    paths = [
        Path(os.environ[name])
        for name in (
            "HARBICHESS_TEST_E8",
            "HARBICHESS_TEST_POSITION",
            "HARBICHESS_TEST_LINEAR",
            "HARBICHESS_TEST_ZERO",
        )
    ]
    board = chess.Board()
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(move)
    encoder = TorchArrayBoardEncoder()
    x = torch.tensor(np.asarray(encoder.encode_board(board).values).reshape(1, 8, 8, 104))
    actions = torch.tensor([legal_action_indices(board)])
    for path in paths:
        model = load_weights(path).eval()
        adapter = NeuralValue(path)
        with torch.inference_mode():
            logits = model.masked_policy_value(x, actions)[1]
            assert logits.numpy().tobytes() == adapter.logits(board).numpy().tobytes()
            win, _, loss = _softmax(tuple(logits[0].tolist()))
            assert adapter(board) == win - loss
