"""PolicyValueBackend adapter for the MLX HarbiChess network."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.core.backend import (
    BackendCapabilities,
    EncodedPosition,
    MaskedPolicyValueOutput,
    PolicyValueOutput,
)


class MLXPolicyValueBackend:
    def __init__(
        self,
        network: HarbiChessNetwork | None = None,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        compiled: bool = True,
        fixed_batch_size: int | None = None,
    ) -> None:
        if fixed_batch_size is not None and fixed_batch_size <= 0:
            raise ValueError("fixed MLX batch size must be positive")
        self.network = network or HarbiChessNetwork()
        self.dtype = dtype
        self.fixed_batch_size = fixed_batch_size
        self.network.set_dtype(dtype)
        self.network.eval()
        mx.eval(self.network.parameters())
        self._compiled = compiled
        self._thread_local = threading.local()
        self._capabilities = BackendCapabilities(
            name="mlx",
            device=mx.device_info()["device_name"],
            supports_training=True,
            supports_compilation=True,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def evaluate(self, positions: Sequence[EncodedPosition]) -> list[PolicyValueOutput]:
        if not positions:
            return []
        policy, wdl = self._forward(positions)
        return [
            PolicyValueOutput(tuple(map(float, policy_row)), tuple(map(float, wdl_row)))
            for policy_row, wdl_row in zip(policy.tolist(), wdl.tolist(), strict=True)
        ]

    def evaluate_masked(
        self,
        positions: Sequence[EncodedPosition],
        action_indices: Sequence[tuple[int, ...]],
    ) -> list[MaskedPolicyValueOutput]:
        if len(positions) != len(action_indices):
            raise ValueError("every position must have one legal action index tuple")
        if not positions:
            return []
        if any(not indices for indices in action_indices):
            raise ValueError("masked policy evaluation requires legal actions")
        policy, wdl = self._forward(positions)
        policy_size = policy.shape[1]
        if any(
            action < 0 or action >= policy_size
            for indices in action_indices
            for action in indices
        ):
            raise ValueError("masked policy action index is out of range")
        maximum = max(map(len, action_indices))
        padded = [
            (*indices, *(0 for _ in range(maximum - len(indices))))
            for indices in action_indices
        ]
        gathered = mx.take_along_axis(policy, mx.array(padded, dtype=mx.int32), axis=1)
        mx.eval(gathered, wdl)
        policy_rows = gathered.tolist()
        return [
            MaskedPolicyValueOutput(
                tuple(map(float, policy_row[: len(indices)])),
                tuple(map(float, wdl_row)),
            )
            for policy_row, wdl_row, indices in zip(
                policy_rows,
                wdl.tolist(),
                action_indices,
                strict=True,
            )
        ]

    def _forward(
        self,
        positions: Sequence[EncodedPosition],
    ) -> tuple[mx.array, mx.array]:
        shape = positions[0].shape
        schema_version = positions[0].schema_version
        if any(
            position.shape != shape or position.schema_version != schema_version
            for position in positions
        ):
            raise ValueError("all positions in an MLX batch must share shape and schema")
        actual_size = len(positions)
        if self.fixed_batch_size is not None and actual_size > self.fixed_batch_size:
            raise ValueError("MLX batch exceeds configured fixed batch size")
        padded_positions = list(positions)
        if self.fixed_batch_size is not None:
            padded_positions.extend(
                positions[-1] for _ in range(self.fixed_batch_size - actual_size)
            )
        inputs = mx.array(
            [position.values for position in padded_positions], dtype=self.dtype
        )
        inputs = inputs.reshape((len(padded_positions), *shape))
        policy, wdl = self._thread_forward()(inputs)
        policy = policy[:actual_size]
        wdl = wdl[:actual_size]
        mx.eval(policy, wdl)
        return policy, wdl

    def _thread_forward(self) -> Callable[[mx.array], tuple[mx.array, mx.array]]:
        if not self._compiled:
            return self.network
        forward = getattr(self._thread_local, "forward", None)
        if forward is None:
            forward = mx.compile(self.network)
            self._thread_local.forward = forward
        return forward
