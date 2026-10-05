"""Versioned offline full-value native payload helpers; scratch CPU tests only."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from core import frozen_bits

NATIVE_SCHEMA = "fresh-qsearch-full-critic-own-search-native-v1"


def _hash_bits(mapping: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, bits in sorted(mapping.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(bits)
    return digest.hexdigest()


def capture_native(model, optimizer, trainable_names, contract, accepted, sampler):
    if accepted < 0 or not isinstance(contract, dict) or not contract:
        raise ValueError("invalid full-critic native cursor/contract")
    names = tuple(trainable_names)
    named = dict(model.named_parameters())
    if not names or any(name not in named or not named[name].requires_grad for name in names):
        raise ValueError("native trainable list does not match model mask")
    frozen = frozen_bits(model, names)
    numpy_state = np.random.get_state()
    return {
        "schema": NATIVE_SCHEMA,
        "contract": contract,
        "accepted": accepted,
        "trainable_names": names,
        "trainable_state": {name: named[name].detach().cpu().clone() for name in names},
        "frozen_parameter_sha256": _hash_bits(frozen),
        "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.getstate(),
        "python_rng": random.getstate(),
        "numpy_rng": {
            "kind": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gauss": numpy_state[4],
        },
        "torch_cpu_rng": torch.get_rng_state().clone(),
    }


def restore_native(payload, model, optimizer, trainable_names, expected_contract, sampler):
    names = tuple(trainable_names)
    if (
        payload.get("schema") != NATIVE_SCHEMA
        or payload.get("contract") != expected_contract
        or payload.get("trainable_names") != names
        or set(payload.get("trainable_state", {})) != set(names)
        or type(payload.get("accepted")) is not int
        or payload["accepted"] < 0
    ):
        raise ValueError("full-critic native schema/contract/mask/cursor differs")
    frozen_hash = _hash_bits(frozen_bits(model, names))
    if frozen_hash != payload.get("frozen_parameter_sha256"):
        raise ValueError("frozen E8 shared/policy/value storage identity differs")
    state = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            target = state[name]
            source = payload["trainable_state"][name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError("full-critic trainable parameter structure differs")
            target.copy_(source)
    optimizer.load_state_dict(payload["optimizer"])
    if payload["accepted"]:
        if len(optimizer.state) != len(names) or any(
            int(state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"])
            != payload["accepted"]
            for state in optimizer.state.values()
        ):
            raise ValueError("Adam step cursor differs from full-critic native counter")
    elif optimizer.state:
        raise ValueError("fresh-Adam native counter/state mismatch")
    sampler.setstate(payload["sampler_rng"])
    random.setstate(payload["python_rng"])
    numpy_state = payload["numpy_rng"]
    np.random.set_state(
        (
            numpy_state["kind"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gauss"],
        )
    )
    torch.set_rng_state(payload["torch_cpu_rng"])
    return payload["accepted"]
