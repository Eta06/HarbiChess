"""Versioned policy-preserving/new-value weights transition, never full resume."""

from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256


def transfer(source: Path, destination: Path, *, channels=128, hidden=64, seed=20261022):
    if destination.exists():
        raise FileExistsError(destination)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old = load_weights(source)
        if (
            old.architecture != "pairwise"
            or old._policy_adapter is not None
            or old._policy_context is not None
            or old._value_sparse is not None
        ):
            raise ValueError("sparse value transfer requires plain versioned pairwise weights")
        target = TorchChessNetwork.from_specification(
            {
                **old.specification,
                "value_sparse": {"schema": 1, "channels": channels, "hidden": hidden},
            }
        )
        incompatible = target.load_state_dict(old.state_dict(), strict=False)
        expected = {name for name in target.state_dict() if name.startswith("value_sparse_head.")}
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
            raise ValueError("sparse value transition changed inherited parameter mapping")
    provenance = {
        "transfer_schema": 1,
        "method": "sparse-value-v1",
        "source_sha256": sha256(source),
        "seed": seed,
        "transfer": "weights-only; policy preserved, NEW RANDOM VALUE, old value inactive",
        "optimizer": "reset; no optimizer migration",
        "full_training_resume": False,
        "initial_value_identity": False,
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "sparse_value_parameters": sum(
            p.numel()
            for name, p in target.named_parameters()
            if name.startswith("value_sparse_head.")
        ),
        "specification": target.specification,
    }
