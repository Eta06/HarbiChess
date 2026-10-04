"""Torch-free MLX counterpart of version-1 policy-only global attention."""

import math

import mlx.core as mx
import mlx.nn as nn


class PolicyContextBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads, self.depth = heads, channels // heads
        self.position = mx.zeros((64, channels))
        self.norm1 = nn.LayerNorm(channels, eps=1e-5)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels, eps=1e-5)
        self.ff1 = nn.Linear(channels, 2 * channels)
        self.ff2 = nn.Linear(2 * channels, channels)
        for layer in (self.proj, self.ff2):
            layer.weight = mx.zeros_like(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, tokens):
        size = tokens.shape[0]
        qkv = self.qkv(self.norm1(tokens + self.position)).reshape(
            size, 64, 3, self.heads, self.depth
        )
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = mx.softmax(q @ mx.swapaxes(k, -2, -1) / math.sqrt(self.depth), axis=-1)
        context = mx.swapaxes(attention @ v, 1, 2).reshape(size, 64, -1)
        tokens = tokens + self.proj(context)
        return tokens + self.ff2(mx.maximum(self.ff1(self.norm2(tokens)), 0))
