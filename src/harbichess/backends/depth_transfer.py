"""Versioned weights-only residual deepening; optimizer state is never transferred."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256


def deepen(source: Path, destination: Path, *, extra_blocks: int = 2, seed: int = 20261009) -> dict:
    if extra_blocks <= 0:
        raise ValueError("extra residual block count must be positive")
    if destination.exists():
        raise FileExistsError(destination)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old = load_weights(source)
        if old.architecture != "pairwise":
            raise ValueError("depth transfer requires versioned pairwise weights")
        invariant = {**old.invariant, "blocks": old.invariant["blocks"] + extra_blocks}
        target = TorchChessNetwork(
            replace(old.config, residual_blocks=old.config.residual_blocks + extra_blocks),
            architecture="pairwise",
            invariant=invariant,
        )
        new_keys = set(target.state_dict()) - set(old.state_dict())
        result = target.load_state_dict(old.state_dict(), strict=False)
        if result.unexpected_keys or set(result.missing_keys) != new_keys:
            raise ValueError("incomplete inherited parameter transfer")
        new_blocks = (
            *target.blocks[old.config.residual_blocks :],
            *target.value_tower_blocks[old.invariant["blocks"] :],
        )
        with torch.no_grad():
            for block in new_blocks:
                block.conv2.weight.zero_()
                block.conv2.bias.zero_()
        if any(not torch.equal(v, target.state_dict()[k]) for k, v in old.state_dict().items()):
            raise ValueError("inherited parameters changed")
    provenance = {
        "transfer_schema": 1,
        "method": "identity-residual-depth-v1",
        "transfer": "weights-only; added residual branches initially zero",
        "optimizer": "reset; not full training resume",
        "source_sha256": sha256(source),
        "seed": seed,
        "extra_blocks_per_tower": extra_blocks,
        "new_parameter_keys": sorted(new_keys),
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "specification": target.specification,
    }
