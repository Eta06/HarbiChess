"""New explicit family bindings; underlying lossless raw-byte codec is unchanged.

Private module instance prevents alteration of any v1 codec import or allowlist.
No Torch/unpickling/network/model inference or file rewriting.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODEC_SHA = "878868940b5bae15c182dce55215320b5f0ba3692b9c5eac26988204832a31d6"
SOURCES = {
    "cpu-own-outcome-shrunk-training-native-v2": "4cae08522ac940746cf9127ca8ce4a6b63a47408",
    "cpu-own-outcome-residual-training-native-v3": "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
}
SEEDS = {
    "cpu-own-outcome-shrunk-training-native-v2": {20262605, 20262606},
    "cpu-own-outcome-residual-training-native-v3": {20262705, 20262706},
}
spec = importlib.util.spec_from_file_location(
    "_private_exact_byte_codec_v2", ROOT / "native_delta_codec.py"
)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
if core.digest((ROOT / "native_delta_codec.py").read_bytes()) != CODEC_SHA:
    raise ValueError("unchanged-generic-codec-SHA-required")


def native_binding(raw_files):
    if tuple(raw_files) != core.FILES or sum(map(len, raw_files.values())) > core.MAX_RAW_BYTES:
        raise ValueError("exact-bounded-three-files")
    checkpoint = json.loads(raw_files["checkpoint.json"])
    contract = checkpoint["contract"]
    schema = checkpoint["schema"]
    if (
        schema not in SOURCES
        or contract["schema"] != schema
        or contract["source_commit"] != SOURCES[schema]
        or contract["seed"] not in SEEDS[schema]
        or checkpoint["accepted"] != contract["max_steps"]
        or checkpoint["accepted"] != 1024
        or contract["initial_e8_sha256"] != core.E8_SHA256
        or contract["device"] != "cpu"
        or contract["new_teacher_labels"] is not False
        or contract["new_selfplay_moves_generated"] != 0
        or set(checkpoint["artifacts"]) != set(core.FILES[:2])
    ):
        raise ValueError("new-fixed-family-native-contract")
    for name in core.FILES[:2]:
        if core.digest(raw_files[name]) != checkpoint["artifacts"][name]:
            raise ValueError("sealed-native-artifact-sha")
    return {
        "schema": schema,
        "source_commit": contract["source_commit"],
        "initial_e8_sha256": core.E8_SHA256,
        "seed": contract["seed"],
        "accepted": checkpoint["accepted"],
        "torch_version": contract["torch_version"],
        "protocol_sha256": contract["protocol_sha256"],
        "helper_sha256": contract["helper_sha256"],
        "feature_helper_sha256": contract["feature_helper_sha256"],
        "journal_sha256": contract["journal_sha256"],
        "complete_original_contract_sha256": core.digest(core.canonical(contract)),
        "value_composition": contract.get("value_composition"),
    }


core.native_binding = native_binding
for name in dir(core):
    if not name.startswith("_") and name != "native_binding":
        globals()[name] = getattr(core, name)
