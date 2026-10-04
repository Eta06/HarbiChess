"""Current piece-square/EP value MLP; old full-history policy remains separate."""

import torch
from torch import nn


class SparseValueHead(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.feature = nn.Linear(832, channels)
        self.metadata = nn.Linear(8, channels, bias=False)
        self.hidden = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, 3)

    def forward(self, inputs: torch.Tensor):
        size = inputs.shape[0]
        pieces = inputs[:, :, :, :12].reshape(size, 768)
        ep = inputs[:, :, :, 101].reshape(size, 64)
        metadata = inputs[:, :, :, 96:].mean(dim=(1, 2))
        features = self.feature(torch.cat((pieces, ep), dim=1)) + self.metadata(metadata)
        features = features.clamp(0, 1)
        return self.output(self.hidden(features).clamp(0, 1))
