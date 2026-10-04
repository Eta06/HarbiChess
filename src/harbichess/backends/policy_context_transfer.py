"""Explicit weights-only context-v1 transition; optimizer must start fresh."""

from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256


def transfer(source: Path, destination: Path, *, heads=4, blocks=2, seed=20261020):
    if destination.exists():
        raise FileExistsError(destination)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = load_weights(source)
        if (
            original.architecture != "pairwise"
            or original._policy_adapter is not None
            or original._policy_context is not None
            or original._value_sparse is not None
        ):
            raise ValueError("context transfer requires plain versioned pairwise model")
        specification = {
            **original.specification,
            "policy_context": {"schema": 1, "blocks": blocks, "heads": heads},
        }
        target = TorchChessNetwork.from_specification(specification)
        incompatible = target.load_state_dict(original.state_dict(), strict=False)
        expected = {
            name for name in target.state_dict() if name.startswith("policy_context_blocks.")
        }
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
            raise ValueError("context transfer changed inherited parameter mapping")
    provenance = {
        "transfer_schema": 1,
        "method": "policy-context-v1",
        "source_sha256": sha256(source),
        "seed": seed,
        "transfer": "weights-only; identity residual context, optimizer reset",
        "full_training_resume": False,
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "context_parameters": sum(
            p.numel()
            for name, p in target.named_parameters()
            if name.startswith("policy_context_blocks.")
        ),
        "specification": specification,
    }
