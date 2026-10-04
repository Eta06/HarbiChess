"""Small identity-initialized ordinary attention for policy tokens only."""

import math

import torch
from torch import nn


class PolicyContextBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads, self.depth = heads, channels // heads
        self.position = nn.Parameter(torch.zeros(64, channels))
        self.norm1 = nn.LayerNorm(channels, eps=1e-5)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels, eps=1e-5)
        self.ff1 = nn.Linear(channels, 2 * channels)
        self.ff2 = nn.Linear(2 * channels, channels)
        for layer in (self.proj, self.ff2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.Tensor):
        size = tokens.shape[0]
        q, k, v = (
            self.qkv(self.norm1(tokens + self.position))
            .reshape(size, 64, 3, self.heads, self.depth)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attention = (q @ k.transpose(-2, -1) / math.sqrt(self.depth)).softmax(-1)
        context = (attention @ v).transpose(1, 2).reshape(size, 64, -1)
        tokens = tokens + self.proj(context)
        return tokens + self.ff2(torch.relu(self.ff1(self.norm2(tokens))))
