"""Own-final-parent admission proposal. Read-only; never a full-resume bridge."""

import hashlib
import json
from pathlib import Path

STATUS = "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength"
END = 1791448916.685839


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def pinned(ref, cap=8 * 2**20):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("exact immutable path/SHA reference required")
    path = Path(ref["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > cap
        or sha(path) != ref["sha256"]
    ):
        raise ValueError("immutable artifact hash/regular-file/size mismatch")
    return path


def read(ref):
    return json.loads(pinned(ref).read_bytes())


def pin_tree(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            pinned({"path": value["path"], "sha256": value["sha256"]})
        for child in value.values():
            pin_tree(child)
    elif isinstance(value, list):
        for child in value:
            pin_tree(child)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def native_rows(result, steps):
    rows = result["native_payloads"]
    if [r["step"] for r in rows] != steps:
        raise ValueError("actual complete fresh-interpreter load inventory required")
    for row in rows:
        pinned({"path": row["path"], "sha256": row["sha256"]})
        if row["actual_fresh_process"] is not True:
            raise ValueError("actual fresh process required")
        log = Path(row["log_path"])
        audits = []
        for owner in result["commands"]:
            cmd = owner["command"]
            if (
                "--audit-only" in cmd
                and "--resume" in cmd
                and cmd[cmd.index("--resume") + 1] == row["path"]
            ):
                audits.append(owner)
        if len(audits) != 1:
            raise ValueError("one actual registered fresh audit command per native")
        owner = audits[0]
        cmd = owner["command"]
        if (
            owner["returncode"] != 0
            or owner["log_sha256"] != sha(log)
            or cmd[cmd.index("--resume-sha256") + 1] != row["sha256"]
            or int(cmd[cmd.index("--stop") + 1]) != row["step"]
            or not result["first"] <= owner["finished"] <= result["deadline"]
        ):
            raise ValueError("original fresh interpreter command/log binding")
        if json.loads(log.read_bytes()) != {
            "status": "PASS-strict-native-readonly",
            "step": row["step"],
        }:
            raise ValueError("strict fresh-load log differs")


def validate_metadata(spec):
    """Hash-bound receipts first. Pure metadata tests do not qualify binary state."""
    if (
        spec["schema"] != "NNUE-own-generation-parent-admission-seal-v3"
        or spec["status"] != "registered"
        or type(spec["generation"]) is not int
        or spec["generation"] < 2
        or spec["weights_only"] is not True
    ):
        raise ValueError("explicit subsequent-generation weights-only seal")
    contract = read(spec["parent_contract"])
    proof_contract = read(spec["parent_proof_contract"])
    proof, fit = read(spec["parent_proof_result"]), read(spec["parent_fit_result"])
    for result, mode, updates, c in [
        (proof, "proof", 8, proof_contract),
        (fit, "fresh-fit", 64, contract),
    ]:
        if (
            result["status"] != STATUS
            or result["mode"] != mode
            or result["own_updates"] != updates
            or result["contract_sha256"]
            != spec["parent_proof_contract" if mode == "proof" else "parent_contract"]["sha256"]
            or not result["first"] < result["finished"] < result["deadline"]
            or result["deadline"] - result["first"] > (600 if mode == "proof" else 1800)
            or result["deadline"] > c["operator_end_epoch"]
            or result["weights_only_initializer"]["sha256"]
            != contract["bootstrap_candidate_sha256"]
        ):
            raise ValueError("closed historical proof/fit clocks and original parent contract")
    if proof["full_payload_bits_equal"] is not True:
        raise ValueError("actual full native pause/resume storage proof required")
    if (
        contract["phase"] != "own-learning"
        or contract["updates"] != 64
        or contract["seed"] != spec["seed"]
        or proof_contract["seed"] != spec["seed"]
        or proof_contract["dataset_sha256"] != contract["dataset_sha256"]
        or proof_contract["bootstrap_candidate_sha256"] != contract["bootstrap_candidate_sha256"]
        or proof_contract["math"] != contract["math"]
        or proof_contract["feature_schema"] != contract["feature_schema"]
        or proof_contract["prior_helper_sha256"] != contract["prior_helper_sha256"]
        or contract["teacher_labels_used_in_own_phase"] is not False
    ):
        raise ValueError("same-data same-initializer own-only parent lineage")
    pin_tree(contract["raw_collection_inputs"])
    generation = contract.get("generation", 1)
    if generation != spec["generation"] - 1:
        raise ValueError("no generation skip or forged teacher-as-own phase")
    native_rows(proof, [0, 8, 0, 4, 4, 8])
    native_rows(fit, [0, 64])
    if spec["parent_native"] != {
        "path": fit["native_payloads"][1]["path"],
        "sha256": fit["native_payloads"][1]["sha256"],
    }:
        raise ValueError("exact final64 native from production fresh-load receipt")
    for key in (
        "parent_candidate",
        "parent_native",
        "parent_native_helper",
        "parent_model_helper",
    ):
        pinned(spec[key])
    if spec["parent_dataset"]["sha256"] != contract["dataset_sha256"]:
        raise ValueError("exact parent own dataset binding")
    pinned(spec["parent_dataset"])
    pinned(
        {
            "path": contract["target_provenance_path"],
            "sha256": contract["target_provenance_sha256"],
        }
    )
    for path, digest in {
        **contract["source_sha256"],
        **contract["execution_helpers_sha256"],
        **contract["inference_source_sha256"],
    }.items():
        pinned({"path": path, "sha256": digest})
    if (
        contract["source_sha256"].get(spec["parent_native_helper"]["path"])
        != spec["parent_native_helper"]["sha256"]
        or contract["source_sha256"].get(spec["parent_model_helper"]["path"])
        != spec["parent_model_helper"]["sha256"]
    ):
        raise ValueError("original native/model helper source closure required")
    return contract


def admit(spec, native, candidate_loader):
    """ROOT may execute later with EXACT original strict loader; no forward/SGD.

    candidate_loader is torch.load(weights_only=False,map_location='cpu') from
    the pinned original closure, not an untrusted/public arbitrary parser.
    """
    contract = validate_metadata(spec)
    if (
        Path(native.__file__).resolve() != pinned(spec["parent_native_helper"]).resolve()
        or sha(native.__file__) != spec["parent_native_helper"]["sha256"]
    ):
        raise ValueError("loaded original strict-native module differs")
    incoming = native.load_native(pinned(spec["parent_native"]), contract)
    packet = candidate_loader(pinned(spec["parent_candidate"]))
    if (
        set(packet) != {"schema", "contract", "model"}
        or packet["schema"] != native.MODEL_SCHEMA
        or packet["contract"] != contract
        or incoming.step != 64
        or not native.bits_equal(packet["model"], incoming.native()["model"])
    ):
        raise ValueError("candidate/full-native original-contract/model storage differs")
    native.validate_weights(packet["model"])
    return packet["model"], {
        "schema": "NNUE-own-generation-parent-admission-result-v3",
        "status": "PASS-own64-parent-native-and-candidate-not-strength",
        "generation": spec["generation"],
        "seed": spec["seed"],
        "parent_contract_canonical_sha256": hashlib.sha256(canonical(contract)).hexdigest(),
        "parent_candidate": spec["parent_candidate"],
        "parent_native": spec["parent_native"],
        "parent_proof_result": spec["parent_proof_result"],
        "parent_fit_result": spec["parent_fit_result"],
        "incoming_native_full_resume_checked": True,
        "outgoing_bridge_full_resume": False,
        "new_baseline": "exact admitted current parent model storage",
        "new_optimizer_and_rng": True,
        "teacher_labels_used_for_new_targets": False,
    }


def validate_admission_result(seal_ref, result_ref):
    spec = read(seal_ref)
    contract = validate_metadata(spec)
    result = read(result_ref)
    clock = read(result["clock"])
    if (
        clock["schema"] != "NNUE-own-parent-readonly-admission-clock-v3"
        or clock["seal_sha256"] != seal_ref["sha256"]
        or result["clock_sha256"] != result["clock"]["sha256"]
        or (result["first"], result["deadline"]) != (clock["first"], clock["deadline"])
        or not result["first"]
        < result["finished"]
        <= result["deadline"]
        <= min(result["first"] + 600, clock["operator_end_epoch"], END)
        or result["schema"] != "NNUE-own-generation-parent-admission-result-v3"
        or result["status"] != "PASS-own64-parent-native-and-candidate-not-strength"
        or result["seal_sha256"] != seal_ref["sha256"]
        or result["bridge_sha256"] != sha(__file__)
        or result["generation"] != spec["generation"]
        or result["seed"] != spec["seed"]
        or result["parent_candidate"] != spec["parent_candidate"]
        or result["parent_native"] != spec["parent_native"]
        or result["parent_contract_canonical_sha256"]
        != hashlib.sha256(canonical(contract)).hexdigest()
        or result["incoming_native_full_resume_checked"] is not True
        or result["outgoing_bridge_full_resume"] is not False
    ):
        raise ValueError("actual sealed strict original-parent admission required")
    return spec, contract


def admitted_candidate(seal_ref, result_ref, torch):
    spec, contract = validate_admission_result(seal_ref, result_ref)
    packet = torch.load(pinned(spec["parent_candidate"]), map_location="cpu", weights_only=False)
    if (
        set(packet) != {"schema", "contract", "model"}
        or packet["schema"] != "own-kingbucket-nnue16-model-v1"
        or packet["contract"] != contract
    ):
        raise ValueError("exact admitted candidate/original contract")
    return packet
