"""NHWC-compatible PyTorch networks and strict versioned MLX weight conversion.

Conv weights are OIHW here, OHWI in MLX. Dense heads retain NHWC flattening.
Conversion transfers model weights only; MLX optimizer state is never discarded
under the name of a training resume. Unknown experimental heads fail strictly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from harbichess.chess.encoding import HISTORY_STEPS, METADATA_PLANES, PIECE_PLANES_PER_STEP
from harbichess.core.network_config import NetworkConfig

WEIGHT_SCHEMA = 1
ARCHITECTURES = ("base", "invariant", "decoupled", "plastic")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class TorchChessNetwork(nn.Module):
    """Port of base, invariant, MIHVER and DENGE production forward paths."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        architecture: str = "base",
        invariant: dict[str, int] | None = None,
        plastic: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.config = config or NetworkConfig()
        self.architecture = architecture
        self.invariant = invariant or {"channels": 16, "blocks": 2, "hidden": 32}
        self.plastic = plastic or {
            "channels": 16,
            "blocks": 2,
            "hidden": 64,
            "invariant_hidden": 32,
        }
        if any(x <= 0 for x in (*self.invariant.values(), *self.plastic.values())):
            raise ValueError("residual dimensions must be positive")
        c = self.config
        self.stem = nn.Conv2d(c.input_channels, c.trunk_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(c.trunk_channels) for _ in range(c.residual_blocks)]
        )
        self.policy_conv = nn.Conv2d(c.trunk_channels, c.policy_channels, 1)
        self.policy_linear = nn.Linear(64 * c.policy_channels, c.policy_size)
        self.value_conv = nn.Conv2d(c.trunk_channels, c.value_channels, 1)
        self.value_hidden = nn.Linear(64 * c.value_channels, c.value_hidden)
        self.value_output = nn.Linear(c.value_hidden, 3)
        features = PIECE_PLANES_PER_STEP + METADATA_PLANES
        if architecture != "base":
            v = self.invariant
            self.invariant_value_linear = nn.Linear(features, 3)
            self.value_tower_stem = nn.Conv2d(c.input_channels, v["channels"], 3, padding=1)
            self.value_tower_blocks = nn.ModuleList(
                [ResidualBlock(v["channels"]) for _ in range(v["blocks"])]
            )
            self.value_tower_hidden = nn.Linear(2 * v["channels"], v["hidden"])
            self.value_tower_output = nn.Linear(v["hidden"], 3)
            self._zero(self.invariant_value_linear, self.value_tower_output)
        if architecture in ("decoupled", "plastic"):
            self.material_value_linear = nn.Linear(features, 1)
            self.global_value_hidden = nn.Linear(features, 64)
            self.global_value_output = nn.Linear(64, 3)
            self._zero(self.material_value_linear, self.global_value_output)
        if architecture == "plastic":
            v = self.plastic
            self.plastic_invariant_hidden = nn.Linear(features, v["invariant_hidden"])
            self.plastic_tower_stem = nn.Conv2d(c.input_channels, v["channels"], 3, padding=1)
            self.plastic_tower_blocks = nn.ModuleList(
                [ResidualBlock(v["channels"]) for _ in range(v["blocks"])]
            )
            self.plastic_value_hidden = nn.Linear(
                v["invariant_hidden"] + 2 * v["channels"], v["hidden"]
            )
            self.plastic_value_output = nn.Linear(v["hidden"], 3)
            self.value_logit_scale = nn.Parameter(torch.zeros(1))
            self._zero(self.plastic_value_output)

    @staticmethod
    def _zero(*layers: nn.Linear) -> None:
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def specification(self) -> dict:
        return {
            "config": asdict(self.config),
            "architecture": self.architecture,
            "invariant": self.invariant,
            "plastic": self.plastic,
        }

    @classmethod
    def from_specification(cls, specification: dict) -> TorchChessNetwork:
        return cls(
            NetworkConfig(**specification["config"]),
            **{k: specification[k] for k in ("architecture", "invariant", "plastic")},
        )

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    @staticmethod
    def _invariants(inputs: torch.Tensor) -> torch.Tensor:
        counts = inputs[:, :, :, :PIECE_PLANES_PER_STEP].sum(dim=(1, 2))
        metadata = inputs[:, :, :, HISTORY_STEPS * PIECE_PLANES_PER_STEP :].mean(dim=(1, 2))
        return torch.cat((counts, metadata), dim=1)

    @staticmethod
    def _tower(x: torch.Tensor, stem: nn.Module, blocks: nn.ModuleList) -> torch.Tensor:
        x = torch.relu(stem(x))
        for block in blocks:
            x = block(x)
        return torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)

    def _features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (8, 8, c.input_channels):
            raise ValueError(f"network input must have shape (batch, 8, 8, {c.input_channels})")
        x = inputs.permute(0, 3, 1, 2).contiguous()
        trunk = torch.relu(self.stem(x))
        for block in self.blocks:
            trunk = block(trunk)
        policy = self._flatten(torch.relu(self.policy_conv(trunk)))
        value = torch.relu(self.value_hidden(self._flatten(torch.relu(self.value_conv(trunk)))))
        logits = self.value_output(value)
        if self.architecture != "base":
            invariants = self._invariants(inputs)
            tower = self._tower(x, self.value_tower_stem, self.value_tower_blocks)
            logits = logits + self.invariant_value_linear(invariants)
            logits = logits + self.value_tower_output(torch.relu(self.value_tower_hidden(tower)))
        if self.architecture in ("decoupled", "plastic"):
            logits = logits + self.global_value_output(
                torch.relu(self.global_value_hidden(invariants))
            )
        if self.architecture == "plastic":
            tower = self._tower(x, self.plastic_tower_stem, self.plastic_tower_blocks)
            global_features = torch.relu(self.plastic_invariant_hidden(invariants))
            hidden = torch.relu(self.plastic_value_hidden(torch.cat((global_features, tower), 1)))
            logits = (logits + self.plastic_value_output(hidden)) * self.value_logit_scale.exp()
        return policy, logits

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy, value = self._features(inputs)
        return self.policy_linear(policy), value

    def masked_policy_value(
        self, inputs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 2 or actions.shape[0] != inputs.shape[0] or actions.shape[1] == 0:
            raise ValueError("masked actions must have shape (batch, non-zero actions)")
        if torch.any(actions < 0) or torch.any(actions >= self.config.policy_size):
            raise ValueError("masked action index out of range")
        features, value = self._features(inputs)
        weights = self.policy_linear.weight[actions]
        logits = (features[:, None, :] * weights).sum(2) + self.policy_linear.bias[actions]
        return logits, value


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_weights(
    path: Path,
    network: TorchChessNetwork,
    *,
    provenance: dict | None = None,
    mlx_layout: bool = False,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    weights = {k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()}
    if mlx_layout:
        weights = {
            k: v.permute(0, 2, 3, 1).contiguous() if v.ndim == 4 else v for k, v in weights.items()
        }
    metadata = {
        "schema": WEIGHT_SCHEMA,
        "layout": "mlx-ohwi" if mlx_layout else "torch-oihw",
        "specification": network.specification,
        "transfer": "weights-only",
        "provenance": provenance or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(fd)
    try:
        save_file(weights, temporary, metadata={"harbichess": json.dumps(metadata)})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publish without overwriting concurrent writers.
    finally:
        os.unlink(temporary)


def _infer_mlx(weights: dict[str, torch.Tensor]) -> TorchChessNetwork:
    def blocks(prefix: str) -> int:
        return len({k.split(".")[1] for k in weights if k.startswith(prefix + ".")})

    c = NetworkConfig(
        input_channels=weights["stem.weight"].shape[-1],
        trunk_channels=weights["stem.weight"].shape[0],
        residual_blocks=blocks("blocks"),
        policy_channels=weights["policy_conv.weight"].shape[0],
        policy_size=weights["policy_linear.weight"].shape[0],
        value_channels=weights["value_conv.weight"].shape[0],
        value_hidden=weights["value_hidden.weight"].shape[0],
    )
    architecture = "base"
    invariant = plastic = None
    if "invariant_value_linear.weight" in weights:
        architecture = "invariant"
        invariant = {
            "channels": weights["value_tower_stem.weight"].shape[0],
            "blocks": blocks("value_tower_blocks"),
            "hidden": weights["value_tower_hidden.weight"].shape[0],
        }
    if "global_value_output.weight" in weights:
        architecture = "decoupled"
    if "plastic_value_output.weight" in weights:
        architecture = "plastic"
        plastic = {
            "channels": weights["plastic_tower_stem.weight"].shape[0],
            "blocks": blocks("plastic_tower_blocks"),
            "hidden": weights["plastic_value_hidden.weight"].shape[0],
            "invariant_hidden": weights["plastic_invariant_hidden.weight"].shape[0],
        }
    return TorchChessNetwork(c, architecture=architecture, invariant=invariant, plastic=plastic)


def load_weights(path: Path, *, legacy_mlx: bool = False) -> TorchChessNetwork:
    with safe_open(path, framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get("harbichess")
    weights = load_file(str(path))
    if raw is None:
        if not legacy_mlx:
            raise ValueError("unversioned weights: explicitly select legacy MLX warm start")
        network = _infer_mlx(weights)
        layout = "mlx-ohwi"
    else:
        metadata = json.loads(raw)
        if metadata["schema"] != WEIGHT_SCHEMA:
            raise ValueError("unsupported weight schema")
        layout = metadata["layout"]
        network = TorchChessNetwork.from_specification(metadata["specification"])
    if layout not in ("mlx-ohwi", "torch-oihw"):
        raise ValueError("unsupported tensor layout")
    if layout == "mlx-ohwi":
        weights = {
            k: v.permute(0, 3, 1, 2).contiguous() if v.ndim == 4 else v for k, v in weights.items()
        }
    network.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(v).all() for v in network.state_dict().values()):
        raise ValueError("non-finite model weights")
    return network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--legacy-mlx", action="store_true")
    parser.add_argument("--export-mlx", action="store_true")
    args = parser.parse_args()
    network = load_weights(args.source, legacy_mlx=args.legacy_mlx)
    save_weights(
        args.destination,
        network,
        mlx_layout=args.export_mlx,
        provenance={"source_sha256": sha256(args.source)},
    )
    print(
        json.dumps(
            {
                "schema": WEIGHT_SCHEMA,
                "transfer": "weights-only",
                "optimizer": "reset",
                "source_sha256": sha256(args.source),
                "destination_sha256": sha256(args.destination),
                "specification": network.specification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
