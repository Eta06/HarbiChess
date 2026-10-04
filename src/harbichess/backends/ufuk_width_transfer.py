"""Versioned weights-only widening with active new features and zero output bridges."""

from dataclasses import replace
from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256


def widen(
    source: Path,
    destination: Path,
    *,
    trunk_channels=64,
    value_tower_channels=32,
    seed=20261017,
) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    if any(type(v) is not int or v <= 0 for v in (trunk_channels, value_tower_channels)):
        raise ValueError("widths must be positive integers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old = load_weights(source)
        if (
            old.architecture != "pairwise"
            or old._policy_adapter is not None
            or old._policy_context is not None
        ):
            raise ValueError("width transfer requires plain versioned pairwise weights")
        c, v = old.config.trunk_channels, old.invariant["channels"]
        if trunk_channels <= c or value_tower_channels <= v:
            raise ValueError("both widths must increase")
        target = TorchChessNetwork(
            replace(old.config, trunk_channels=trunk_channels),
            architecture="pairwise",
            invariant={**old.invariant, "channels": value_tower_channels},
            plastic=old.plastic,
        )
        old_state, new_state = old.state_dict(), target.state_dict()
        if old_state.keys() != new_state.keys():
            raise ValueError("width transfer changed parameter names")
        with torch.no_grad():
            for name, before in old_state.items():
                after = new_state[name]
                if before.shape == after.shape:
                    after.copy_(before)
                elif name == "pair_hidden.weight":
                    after.zero_()
                    after[:, :c].copy_(before[:, :c])
                    after[:, trunk_channels:].copy_(before[:, c:])
                elif name == "value_tower_hidden.weight":
                    after.zero_()
                    after[:, :v].copy_(before[:, :v])
                    after[:, value_tower_channels : value_tower_channels + v].copy_(before[:, v:])
                elif name == "value_conv.weight":
                    after.zero_()
                    after[:, :c].copy_(before)
                elif name.startswith(("stem.", "value_tower_stem.")):
                    after[: before.shape[0]].copy_(before)
                elif name.startswith(("blocks.", "value_tower_blocks.")):
                    if before.ndim == 4:
                        after[: before.shape[0], before.shape[1] :].zero_()
                        after[: before.shape[0], : before.shape[1]].copy_(before)
                    elif before.ndim == 1:
                        after[: before.shape[0]].copy_(before)
                    else:
                        raise ValueError(f"unrecognized residual parameter {name}")
                else:
                    raise ValueError(f"unrecognized width-dependent parameter {name}")
    provenance = {
        "transfer_schema": 1,
        "method": "active-feature-width-v1",
        "transfer": "weights-only; initial function preserved mathematically, floating tolerance",
        "optimizer": "reset; no optimizer transfer or full training resume",
        "source_sha256": sha256(source),
        "seed": seed,
        "old_widths": [c, v],
        "new_widths": [trunk_channels, value_tower_channels],
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "specification": target.specification,
    }
