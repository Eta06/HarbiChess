"""Isolated CPU/CUDA inference adapter for the shared chess/search contracts."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.encoding import ENCODER_SCHEMA_VERSION
from harbichess.core.backend import (
    BackendCapabilities,
    EncodedPosition,
    MaskedPolicyValueOutput,
    PolicyValueOutput,
)


class TorchPolicyValueBackend:
    """Own an inference snapshot; never mutate a learner's dtype or training mode."""

    def __init__(
        self, network: TorchChessNetwork, *, device: str = "cpu", compiled: bool = False
    ) -> None:
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("PyTorch backend supports cpu or cuda; use MLX on Apple Silicon")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = selected
        self.network = copy.deepcopy(network).to(device=selected, dtype=torch.float32).eval()
        self.network.requires_grad_(False)
        if compiled and (selected.type != "cpu" or network.architecture != "pairwise"):
            raise ValueError("compiled masked inference supports CPU pairwise networks only")
        self._compiled_masked = (
            torch.compile(
                self.network._features,
                backend="inductor",
                fullgraph=True,
                dynamic=True,
                options={"compile_threads": 1},
            )
            if compiled
            else None
        )
        self.capabilities = BackendCapabilities("torch", str(selected), True, compiled)

    def _masked(self, inputs: torch.Tensor, actions: torch.Tensor):
        if self._compiled_masked is None:
            return self.network.masked_policy_value(inputs, actions)
        # Keep the same host guards as the validated eager boundary. Only the
        # numeric pairwise body is compiled; compiler failures never fall back.
        if actions.ndim != 2 or actions.shape[0] != inputs.shape[0] or actions.shape[1] == 0:
            raise ValueError("masked actions must have shape (batch, non-zero actions)")
        if torch.any(actions < 0) or torch.any(actions >= self.network.config.policy_size):
            raise ValueError("masked action outside policy range")
        return self._compiled_masked(inputs, actions)

    def _inputs(self, positions: Sequence[EncodedPosition]) -> torch.Tensor:
        shape = (8, 8, self.network.config.input_channels)
        if any(p.shape != shape or p.schema_version != ENCODER_SCHEMA_VERSION for p in positions):
            raise ValueError("encoded position shape/schema incompatible with backend")
        return torch.tensor(
            [p.values for p in positions], dtype=torch.float32, device=self.device
        ).reshape(len(positions), *shape)

    def evaluate(self, positions: Sequence[EncodedPosition]) -> list[PolicyValueOutput]:
        if not positions:
            return []
        with torch.inference_mode():
            policy, wdl = self.network(self._inputs(positions))
        return [
            PolicyValueOutput(tuple(p), tuple(v))
            for p, v in zip(policy.cpu().tolist(), wdl.cpu().tolist(), strict=True)
        ]

    def evaluate_masked(
        self, positions: Sequence[EncodedPosition], action_indices: Sequence[tuple[int, ...]]
    ) -> list[MaskedPolicyValueOutput]:
        if len(positions) != len(action_indices):
            raise ValueError("one action tuple required per position")
        if not positions:
            return []
        if any(
            not actions or any(a < 0 or a >= self.network.config.policy_size for a in actions)
            for actions in action_indices
        ):
            raise ValueError("masked actions must be nonempty and in range")
        width = max(map(len, action_indices))
        padded = [(*actions, *([0] * (width - len(actions)))) for actions in action_indices]
        with torch.inference_mode():
            policy, wdl = self._masked(
                self._inputs(positions), torch.tensor(padded, dtype=torch.long, device=self.device)
            )
        return [
            MaskedPolicyValueOutput(tuple(p[: len(actions)]), tuple(v))
            for p, v, actions in zip(
                policy.cpu().tolist(), wdl.cpu().tolist(), action_indices, strict=True
            )
        ]
