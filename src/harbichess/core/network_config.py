"""Framework-neutral network dimensions; tensor and action schemas stay shared."""

from dataclasses import dataclass, fields

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS


def validate_policy_adapter(specification: dict | None) -> dict | None:
    """An optional, versioned policy-only branch; never modifies value features."""
    if specification is None:
        return None
    if (
        not isinstance(specification, dict)
        or set(specification) != {"schema", "blocks"}
        or type(specification["schema"]) is not int
        or specification["schema"] != 1
        or type(specification["blocks"]) is not int
        or specification["blocks"] <= 0
    ):
        raise ValueError("unsupported policy adapter specification")
    return dict(specification)


def validate_policy_context(specification: dict | None, channels: int) -> dict | None:
    """Version-1 global policy context; no value or input-schema modification."""
    if specification is None:
        return None
    if (
        not isinstance(specification, dict)
        or set(specification) != {"schema", "blocks", "heads"}
        or any(type(specification[key]) is not int for key in specification)
        or specification["schema"] != 1
        or min(specification["blocks"], specification["heads"]) <= 0
        or channels % specification["heads"]
    ):
        raise ValueError("unsupported policy context specification")
    return dict(specification)


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
