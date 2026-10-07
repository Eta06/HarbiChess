"""ROOT-clocked own64 contracts; cached original teacher admission, no optimizer step."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from convert import canonical, module, pinned, pins_tree, sha

HERE = Path(__file__).resolve().parent


def clock(spec):
    first, end, operator = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    cap = 600 if spec["mode"] == "proof" else 1800
    if (spec["mode"] not in ("proof", "fresh-fit")
            or any(type(x) not in (float, int) or not math.isfinite(x)
                   for x in (first, end, operator))
            or not first < end <= min(first + cap, operator)
            or operator > 1791448916.685839):
        raise ValueError("ROOT new prospective phase clocks; no historical clock mutation")


def build(spec):
    if spec["schema"] != "NNUE-own-contract-build-seal-v2" or spec["status"] != "registered":
        raise ValueError("ROOT immutable contract build seal")
    clock(spec)
    data = pinned(spec["dataset"])
    provenance_path = pinned(spec["provenance"])
    provenance = json.loads(provenance_path.read_bytes())
    rows = json.loads(data.read_bytes())
    if (rows["schema"] != "own-kingbucket-sparse-training-data-v1"
            or rows["phase"] != "own-learning" or len(rows["rows"]) != 1024
            or provenance["schema"] != "NNUE-own1024-converted-data-provenance-v1"
            or provenance["teacher_labels_used"] is not False
            or provenance["dataset_sha256"] != spec["dataset"]["sha256"]
            or provenance["seed"] != spec["seed"]):
        raise ValueError("strict1024 own-data and converted provenance")
    pins_tree(provenance["inputs"])
    conversion = provenance["inputs"]
    receipt = json.loads(pinned(conversion["receipt"]).read_bytes())
    if receipt["parent_candidate_sha256"] != spec["teacher"]["candidate"]["sha256"]:
        raise ValueError("exact current teacher parent produces own targets")
    sys.path.insert(0, str(HERE))
    native = module(dict(path=str(HERE / "native.py"), sha256=sha(HERE / "native.py")),
                    "own_contract_native")
    original_admission = module(spec["teacher_admission"], "cached_teacher_admission")
    _, _, teacher_proof = original_admission.teacher(spec["teacher"], native)
    parent_contract = json.loads(pinned(spec["teacher"]["contract"]).read_bytes())
    collection_reg = json.loads(pinned(conversion["registration"]).read_bytes())
    parent_binding = collection_reg["parent_candidate"]
    expected_parent_contract_sha = hashlib.sha256(canonical(parent_contract)).hexdigest()
    if (parent_binding["contract_sha256"] != expected_parent_contract_sha
            or receipt["parent_contract_sha256"] != expected_parent_contract_sha
            or parent_binding["path"] != spec["teacher"]["candidate"]["path"]):
        raise ValueError("same full teacher contract produces own targets")
    if spec["seed"] != parent_contract["seed"]:
        raise ValueError("same named teacher seed")
    inference = dict(parent_contract["inference_source_sha256"])
    for path, h in inference.items():
        pinned(dict(path=path, sha256=h))
    own_proof = None
    if spec["mode"] == "fresh-fit":
        own_proof = json.loads(pinned(spec["own_proof_result"]).read_bytes())
        if (own_proof["status"] != "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength"
                or own_proof["mode"] != "proof" or own_proof["own_updates"] != 8
                or own_proof["full_payload_bits_equal"] is not True
                or own_proof["weights_only_initializer"]["sha256"] !=
                spec["teacher"]["candidate"]["sha256"]
                or [x["step"] for x in own_proof["native_payloads"]] != [0, 8, 0, 4, 4, 8]
                or not own_proof["first"] < own_proof["finished"] < own_proof["deadline"]):
            raise ValueError("actual own proof before production admission")
        for item in own_proof["native_payloads"]:
            pinned(item)
            log = Path(item["log_path"])
            if json.loads(log.read_bytes()) != dict(status="PASS-strict-native-readonly",
                                                    step=item["step"]):
                raise ValueError("six actual own fresh-load results")
        proof_contract = json.loads(pinned(spec["own_proof_contract"]).read_bytes())
        if (own_proof["contract_sha256"] != spec["own_proof_contract"]["sha256"]
                or proof_contract["dataset_sha256"] != sha(data)
                or proof_contract["bootstrap_candidate_sha256"] !=
                spec["teacher"]["candidate"]["sha256"]
                or proof_contract["seed"] != spec["seed"]
                or proof_contract["math"] != native.MATH):
            raise ValueError("same-data same-parent actual own proof")
        inference[spec["own_proof_result"]["path"]] = spec["own_proof_result"]["sha256"]
    source = {str(HERE / n): sha(HERE / n) for n in ("model.py", "native.py", "train.py")}
    helpers = {str(HERE / n): sha(HERE / n) for n in ("convert.py", "contracts.py", "prove.py")}
    return dict(
        phase="own-learning", updates=64, seed=spec["seed"], math=native.MATH,
        feature_schema=parent_contract["feature_schema"], dataset_sha256=sha(data),
        source_sha256=source, execution_helpers_sha256=helpers,
        execution_scope_schema="NNUE-own-execution-contract-v2", execution_mode=spec["mode"],
        original_first_epoch=spec["first"], original_deadline_epoch=spec["deadline"],
        operator_end_epoch=spec["operator_end_epoch"],
        core_source_repo=parent_contract["core_source_repo"],
        core_source_commit=parent_contract["core_source_commit"],
        target_provenance_path=str(provenance_path), target_provenance_sha256=sha(provenance_path),
        prior_helper_path=parent_contract["prior_helper_path"],
        prior_helper_sha256=parent_contract["prior_helper_sha256"],
        inference_source_sha256=inference,
        bootstrap_candidate_path=str(pinned(spec["teacher"]["candidate"]).resolve()),
        bootstrap_candidate_sha256=spec["teacher"]["candidate"]["sha256"],
        weights_bridge="teacher MODEL weights only; NEW Adam/Python/Torch/private sampler RNG",
        teacher_ancestry=spec["teacher"], teacher_admission=spec["teacher_admission"],
        teacher_admission_result=teacher_proof, teacher_labels_used_in_own_phase=False,
        own_phase_proof=spec.get("own_proof_result"),
        collection_receipt_sha256=conversion["receipt"]["sha256"],
        raw_collection_inputs=conversion, contract_build_seal=spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    c = build(json.loads(a.seal.read_bytes()))
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM contract publish only")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        f.write(canonical(c) + b"\n")
