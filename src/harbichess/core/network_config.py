"""Framework-neutral network dimensions; tensor and action schemas stay shared."""

from dataclasses import dataclass, fields

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    input_channels: int = ENCODER_CHANNELS
    trunk_channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 8
    value_channels: int = 4
    value_hidden: int = 64
    policy_size: int = POLICY_SIZE

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
