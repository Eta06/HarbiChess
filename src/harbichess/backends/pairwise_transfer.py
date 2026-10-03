"""Explicit weights-only dense-to-pairwise transfer; optimizer/policy are reset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256


def transfer(source: Path, destination: Path, *, seed: int = 20261003) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    old = load_weights(source)
    if old.architecture != "decoupled":
        raise ValueError("v1 pairwise transfer requires a versioned decoupled source")
    target = TorchChessNetwork(old.config, architecture="pairwise", invariant=old.invariant)
    inherited = {
        k: v
        for k, v in old.state_dict().items()
        if not k.startswith(("policy_conv.", "policy_linear."))
    }
    result = target.load_state_dict(inherited, strict=False)
    if result.unexpected_keys or any(not k.startswith("pair_") for k in result.missing_keys):
        raise ValueError("incomplete trunk/value transfer")
    if any(not torch.equal(v, target.state_dict()[k]) for k, v in inherited.items()):
        raise ValueError("inherited weights changed during transfer")
    provenance = {
        "transfer_schema": 1,
        "transfer": "weights-only architecture transfer",
        "source_sha256": sha256(source),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy": "new initialization",
        "optimizer": "reset",
        "seed": seed,
    }
    save_weights(destination, target, provenance=provenance)
    return {
        **provenance,
        "destination_sha256": sha256(destination),
        "parameters": sum(p.numel() for p in target.parameters()),
        "pair_parameters": sum(
            p.numel() for k, p in target.named_parameters() if k.startswith("pair_")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(transfer(args.source, args.destination)), flush=True)


if __name__ == "__main__":
    main()
