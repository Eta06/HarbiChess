"""Torch-free MLX counterpart of the sparse current-position WDL head."""

import mlx.core as mx
import mlx.nn as nn


class SparseValueHead(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.feature = nn.Linear(832, channels)
        self.metadata = nn.Linear(8, channels, bias=False)
        self.hidden = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, 3)

    def __call__(self, inputs: mx.array):
        size = inputs.shape[0]
        pieces = inputs[:, :, :, :12].reshape(size, 768)
        ep = inputs[:, :, :, 101].reshape(size, 64)
        metadata = mx.mean(inputs[:, :, :, 96:], axis=(1, 2))
        features = self.feature(mx.concatenate((pieces, ep), axis=1)) + self.metadata(metadata)
        features = mx.clip(features, 0, 1)
        return self.output(mx.clip(self.hidden(features), 0, 1))
