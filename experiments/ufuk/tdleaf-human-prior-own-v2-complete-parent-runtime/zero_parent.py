"""Literal zero-output parent metadata; no teacher or old-phase relabel."""

import argparse
import json
from pathlib import Path

from parent_bridge import canonical, pin_tree, pinned, read, sha

PHASE = "human-prior-zero-residual-init-v1"


def zero_head(state):
    import torch

    for key in ("head.weight", "head.bias"):
        if torch.count_nonzero(state[key].reshape(-1).view(torch.uint8)):
            raise ValueError("literal positive-zero output head storage required")


def origin(contract):
    lineage = contract["lineage_origin"]
    if (
        lineage["phase"] != PHASE
        or lineage["seed"] != contract["seed"]
        or lineage["prior_sha256"] != contract["prior_helper_sha256"]
        or lineage["teacher_labels_used"] is not False
    ):
        raise ValueError("typed teacher-free origin required; never teacher256/own64 relabel")


def validate_zero_metadata(spec, contract):
    origin(contract)
    result = read(spec["parent_initialization_result"])
    if (
        spec["generation"] != 1
        or contract["generation"] != 0
        or contract["updates"] != 0
        or contract["phase"] != PHASE
        or contract["seed"] != spec["seed"]
        or result["schema"] != "human-prior-zero-initialization-result-v1"
        or result["status"] != "PASS-literal-zero-parent-and-two-fresh-native-loads-not-strength"
        or result["contract_sha256"] != spec["parent_contract"]["sha256"]
        or result["candidate"] != spec["parent_candidate"]
        or result["native"] != spec["parent_native"]
        or result["source_sha256"] != contract["source_sha256"]
        or result["teacher_labels_used"] is not False
        or result["optimizer_updates"] != 0
        or not result["first"] < result["finished"] < result["deadline"]
        or result["deadline"] > min(result["first"] + 600, contract["operator_end_epoch"])
        or contract["execution_scope_schema"] != "human-prior-zero-initialization-contract-v1"
    ):
        raise ValueError("actual closed literal-zero initialization receipt")
    if [r["step"] for r in result["native_payloads"]] != [0, 0]:
        raise ValueError("two actual fresh native0 loads required")
    if len(result["commands"]) != 2:
        raise ValueError("exact two actual initializer audit commands")
    initializer_paths = [p for p in contract["source_sha256"] if Path(p).name == "initialize.py"]
    if len(initializer_paths) != 1:
        raise ValueError("one exact initializer source closure")
    for row, owner in zip(result["native_payloads"], result["commands"], strict=True):
        cmd = owner["command"]
        log = Path(row["log_path"])
        if (
            row["actual_fresh_process"] is not True
            or cmd[1] != initializer_paths[0]
            or len(cmd) != 9
            or owner["returncode"] != 0
            or row["path"] != spec["parent_native"]["path"]
            or row["sha256"] != spec["parent_native"]["sha256"]
            or "--audit-only" not in cmd
            or cmd[cmd.index("--native") + 1] != row["path"]
            or cmd[cmd.index("--native-sha256") + 1] != row["sha256"]
            or cmd[cmd.index("--contract") + 1] != spec["parent_contract"]["path"]
            or sha(log) != owner["log_sha256"]
            or json.loads(log.read_bytes()) != dict(status="PASS-strict-native-readonly", step=0)
            or not result["first"] <= owner["first"] <= owner["finished"] <= result["deadline"]
        ):
            raise ValueError("actual fresh native0 argv/log/state binding")
    for key in (
        "parent_candidate",
        "parent_native",
        "parent_native_helper",
        "parent_model_helper",
    ):
        pinned(spec[key])
    for path, digest in {
        **contract["source_sha256"],
        **contract["inference_source_sha256"],
    }.items():
        pinned(dict(path=path, sha256=digest))
    pin_tree(contract["initializer_seal"])
    if (
        contract["source_sha256"].get(spec["parent_native_helper"]["path"])
        != spec["parent_native_helper"]["sha256"]
        or contract["source_sha256"].get(spec["parent_model_helper"]["path"])
        != spec["parent_model_helper"]["sha256"]
    ):
        raise ValueError("original initializer native/model source binding")
    return contract


def make_seal(result_ref):
    result = read(result_ref)
    contract_ref = result["contract"]
    contract = read(contract_ref)

    def original_helper(name):
        matches = [path for path in contract["source_sha256"] if Path(path).name == name]
        if len(matches) != 1:
            raise ValueError("unique ORIGINAL zero-parent helper source required")
        return dict(path=matches[0], sha256=contract["source_sha256"][matches[0]])

    spec = dict(
        schema="human-prior-own-parent-admission-seal-v1",
        status="registered",
        generation=1,
        seed=contract["seed"],
        weights_only=True,
        parent_contract=contract_ref,
        parent_initialization_result=result_ref,
        parent_candidate=result["candidate"],
        parent_native=result["native"],
        parent_native_helper=original_helper("native.py"),
        parent_model_helper=original_helper("model.py"),
    )
    validate_zero_metadata(spec, contract)
    return spec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal = make_seal(
        dict(
            path=str(a.initialization_result.resolve()),
            sha256=sha(a.initialization_result),
        )
    )
    with a.output.open("xb") as f:
        f.write(canonical(seal) + b"\n")
    print(json.dumps(dict(path=str(a.output), sha256=sha(a.output))))
