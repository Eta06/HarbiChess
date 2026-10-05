"""New explicit binding adapter; exact oldcodec algorithms remain unchanged."""

import hashlib
import importlib.util
import json
from pathlib import Path

CODEC_SHA = "878868940b5bae15c182dce55215320b5f0ba3692b9c5eac26988204832a31d6"
SOURCE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
FILES = ("candidate.safetensors", "training.pt", "checkpoint.json")
ALLOWED = {
    "fresh-qsearch-additive-own-mc-native-v2",
    "fresh-qsearch-additive-own-sc-native-v1",
    "fresh-qsearch-full-critic-own-search-native-v1",
}


def binding(raw):
    if tuple(raw) != FILES:
        raise ValueError("explicit-exact-three-original-names")
    m = json.loads(raw["checkpoint.json"])
    c = m["contract"]
    if m["schema"] not in ALLOWED or c["source_commit"] != SOURCE or c["device"] != "cpu":
        raise ValueError("explicit-fresh-native-schema-source-device")
    e8 = c.get("frozen_e8_sha256", c.get("e8_sha256"))
    if e8 != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03":
        raise ValueError("exact-e8-external-backbone")
    expected = m.get("artifacts", {}).get("training.pt", m.get("training_pt_sha256"))
    if hashlib.sha256(raw["training.pt"]).hexdigest() != expected:
        raise ValueError("original-native-training-byte-sha")
    if c["seed"] not in (20262805, 20262806) or m["accepted"] != (
        105 if c["seed"] == 20262805 else 97
    ):
        raise ValueError("fixed-final-seed-counter")
    return {
        "schema": m["schema"],
        "source_commit": SOURCE,
        "e8_sha256": e8,
        "seed": c["seed"],
        "accepted": m["accepted"],
        "original_contract": c,
        "contract_sha256": hashlib.sha256(
            json.dumps(c, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }


def codec_instance():
    path = Path(__file__).with_name("native_delta_codec.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != CODEC_SHA:
        raise ValueError("unchanged-codec-source-sha")
    spec = importlib.util.spec_from_file_location("private_fresh_native_delta_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FILES = FILES
    module.SCHEMA = "fresh-six-final-native-exact-delta-v4"
    module.CHUNK_SCHEMA = "fresh-six-final-native-exact-delta-chunks-v4"
    module.native_binding = binding
    return module
