"""Identity policy branch transfer; inherited value features remain independent."""

from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.core.network_config import validate_policy_adapter


def add_policy_adapter(
    source: Path, destination: Path, *, blocks: int = 2, seed: int = 20261012
) -> dict:
    specification = validate_policy_adapter({"schema": 1, "blocks": blocks})
    if destination.exists():
        raise FileExistsError(destination)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old = load_weights(source)
        if old.architecture != "pairwise" or "policy_adapter" in old.specification:
            raise ValueError("adapter transfer requires plain versioned pairwise weights")
        target = TorchChessNetwork.from_specification(
            {**old.specification, "policy_adapter": specification}
        )
        transferred = target.load_state_dict(old.state_dict(), strict=False)
        new_keys = set(target.state_dict()) - set(old.state_dict())
        if (
            transferred.unexpected_keys
            or set(transferred.missing_keys) != new_keys
            or any(not k.startswith("policy_adapter_blocks.") for k in new_keys)
            or any(not torch.equal(v, target.state_dict()[k]) for k, v in old.state_dict().items())
        ):
            raise ValueError("incomplete inherited parameter transfer")
    provenance = {
        "transfer_schema": 1,
        "method": "identity-policy-residual-v1",
        "transfer": "weights-only",
        "optimizer": "reset; not full training resume",
        "source_sha256": sha256(source),
        "seed": seed,
        "policy_adapter": specification,
        "new_parameter_keys": sorted(new_keys),
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "specification": target.specification,
    }
