"""Small JSON native contract skeleton, NOT an implemented/qualified actor CLI.

A future collector must fill every field from actual state. No old v2/E8-native
conversion, default RNG, inferred continuation or optimizer reset is accepted.
"""

import gzip
import hashlib
import json
import random

SCHEMA = "classical-own-actor-native-v3"


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def validate(state, expected_config):
    required = {
        "schema",
        "config",
        "config_sha256",
        "model",
        "frozen_epoch_model_sha256",
        "rng",
        "actions",
        "games",
        "active",
        "epoch_index",
        "journal_parent_sha256",
        "training_native_sha256",
    }
    if set(state) != required or state["schema"] != SCHEMA:
        raise ValueError("complete explicit classical v3 state required")
    if state["config"] != expected_config or state["config_sha256"] != digest(expected_config):
        raise ValueError("immutable config differs")
    if digest(state["model"]) != state["frozen_epoch_model_sha256"]:
        raise ValueError("frozen whole-epoch model differs")
    if not 0 <= state["actions"] <= expected_config["max_actions"]:
        raise ValueError("action count outside prospective budget")
    for key in ("journal_parent_sha256", "training_native_sha256"):
        v = state[key]
        if v is not None and (not isinstance(v, str) or len(v) != 64):
            raise ValueError("explicit parent binding required")
    # JSON preserves all Random state including optional gauss cache.
    rng = random.Random()
    version, words, gauss = state["rng"]
    rng.setstate((version, tuple(words), gauss))
    return rng


def encode(state, config):
    validate(state, config)
    return gzip.compress(canonical({"state": state, "state_sha256": digest(state)}), mtime=0)


def decode(blob, config):
    payload = json.loads(gzip.decompress(blob))
    if digest(payload["state"]) != payload["state_sha256"]:
        raise ValueError("native byte content digest differs")
    validate(payload["state"], config)
    return payload["state"]
