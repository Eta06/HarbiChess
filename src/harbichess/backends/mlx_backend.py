"""PolicyValueBackend adapter for the MLX HarbiChess network."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.core.backend import (
    BackendCapabilities,
    EncodedPosition,
    PolicyValueOutput,
)


class MLXPolicyValueBackend:
    def __init__(
        self,
        network: HarbiChessNetwork | None = None,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        compiled: bool = True,
    ) -> None:
        self.network = network or HarbiChessNetwork()
        self.dtype = dtype
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
        shape = positions[0].shape
        schema_version = positions[0].schema_version
        if any(
            position.shape != shape or position.schema_version != schema_version
            for position in positions
        ):
            raise ValueError("all positions in an MLX batch must share shape and schema")
        inputs = mx.array([position.values for position in positions], dtype=self.dtype)
        inputs = inputs.reshape((len(positions), *shape))
        policy, wdl = self._thread_forward()(inputs)
        mx.eval(policy, wdl)
        return [
            PolicyValueOutput(tuple(map(float, policy_row)), tuple(map(float, wdl_row)))
            for policy_row, wdl_row in zip(policy.tolist(), wdl.tolist(), strict=True)
        ]

    def _thread_forward(self) -> Callable[[mx.array], tuple[mx.array, mx.array]]:
        if not self._compiled:
            return self.network
        forward = getattr(self._thread_local, "forward", None)
        if forward is None:
            forward = mx.compile(self.network)
            self._thread_local.forward = forward
        return forward
