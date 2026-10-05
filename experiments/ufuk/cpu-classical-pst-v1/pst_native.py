"""Strict, versioned full-state checkpoint for the PST offline learner."""

import gzip
import hashlib
import json
import math
import random

from pst_learner import PARAM_COUNT
from pst_value import model_dict

SCHEMA = "classical-own-pst-offline-native-v1"
FIELDS = {
    "schema",
    "contract",
    "theta",
    "m",
    "v",
    "step",
    "candidate",
    "sampler_rng",
    "global_rng",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate(state, contract):
    if (
        set(state) != FIELDS
        or state["schema"] != SCHEMA
        or state["contract"] != contract
    ):
        raise ValueError("exact complete PST native schema/contract required")
    if type(state["step"]) is not int or not 0 <= state["step"] <= contract["updates"]:
        raise ValueError("PST native counter differs")
    for key in ("theta", "m", "v"):
        values = state[key]
        if len(values) != PARAM_COUNT or not all(
            math.isfinite(float(x)) for x in values
        ):
            raise ValueError("PST native vector shape/value differs")
    if any(float(x) < 0 for x in state["v"]):
        raise ValueError("PST Adam second moment negative")
    if state["candidate"] != model_dict(state["theta"][:18], state["theta"][18:]):
        raise ValueError("PST candidate differs from native parameters")
    for name in ("sampler_rng", "global_rng"):
        value = state[name]
        version, words, gaussian = value
        rng = random.Random()
        rng.setstate((version, tuple(words), gaussian))
    return True


def encode(state, contract):
    validate(state, contract)
    payload = {"state": state, "state_sha256": digest(state)}
    return gzip.compress(canonical(payload), mtime=0)


def decode(data, contract):
    payload = json.loads(gzip.decompress(data))
    if digest(payload["state"]) != payload.get("state_sha256"):
        raise ValueError("PST native payload checksum differs")
    validate(payload["state"], contract)
    return payload["state"]
