"""Read-only final64 and fresh0 admission; expired original TRAIN clocks stay stored."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from value import load, read_gzip, sha


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def admit(protocol, seed):
    conversion = protocol["inference_version_conversion"]
    if conversion["schema"] != "C18-inference-reader-v1-to-v2-no-training-state-migration-v1":
        raise ValueError("explicit inference version conversion required")
    for path, expected in conversion["new_inference_source_sha256"].items():
        if sha(path) != expected:
            raise ValueError("new inference conversion closure changed")
    row = protocol["models"][str(seed)]["learned"]
    fit = protocol["C18_fit_inputs"][str(seed)]
    for path_key, sha_key in [("path", "sha256"), ("native_path", "native_sha256"),
                              ("fresh0_path", "fresh0_sha256")]:
        if sha(row[path_key]) != row[sha_key]:
            raise ValueError("sealed candidate/native SHA differs")
    reg = json.loads(Path(fit["registration_path"]).read_bytes())
    if sha(fit["registration_path"]) != fit["registration_sha256"]:
        raise ValueError("fit registration changed")
    candidate, final, zero = [read_gzip(row[k]) for k in ["path", "native_path", "fresh0_path"]]
    contract = final["contract"]
    if sha(reg["labels_path"]) != reg["labels_sha256"]:
        raise ValueError("actual own-label input changed")
    fit_result = json.loads(Path(fit["fit_result_path"]).read_bytes())
    if (sha(fit["fit_result_path"]) != fit["fit_result_sha256"]
            or fit_result.get("schema") != "own-search-residual18-fit-result-v1"
            or fit_result.get("seed") != seed
            or fit_result.get("registration_sha256") != fit["registration_sha256"]
            or fit_result.get("final_native_sha256") != row["native_sha256"]
            or fit_result.get("candidate_sha256") != row["sha256"]
            or fit_result.get("fixed_final_step") != 64
            or fit_result.get("fresh_initializer_step") != 0
            or fit_result.get("training_contract_sha256")
            != hashlib.sha256(canonical(contract)).hexdigest()
            or fit_result.get("original_first_epoch")
            != reg["clocks"]["fit"]["original_first_epoch"]
            or fit_result.get("original_deadline_epoch")
            != reg["clocks"]["fit"]["original_deadline_epoch"]):
        raise ValueError("completed same-scope fit receipt required")
    proof_path = Path(reg["proof_result_path"])
    proof = json.loads(proof_path.read_bytes())
    if (sha(proof_path) != reg["proof_result_sha256"]
            or proof.get("qualification_passed") is not True
            or proof.get("fresh_process_resume") is not True
            or len(proof.get("strict_native_load_sha256", [])) != 6
            or proof.get("seed") != seed
            or proof.get("training_contract_sha256")
            != hashlib.sha256(canonical(contract)).hexdigest()):
        raise ValueError("exact actual native qualification required")
    expected = {
        "source_commit": protocol["source_commit"], "updates": 64,
        "input_count": 18, "hidden_count": 32,
        "labels_sha256": reg["labels_sha256"], "training_code_sha256": reg["closure"],
        "helper_source_commit": reg["helper_source_commit"], "helper_hashes": reg["helper_hashes"],
        "inference_closure": reg["inference_closure"], "prior_sha256": reg["prior_sha256"],
        "training_dataset_sha256": proof["training_dataset_sha256"],
        "actor_config_sha256": reg["actor_config_sha256"],
        "actor_journal_sha256": reg["actor_journal_sha256"],
        "reanalysis_registration_sha256": reg["reanalysis_registration_sha256"],
        "feature_helper_sha256": reg["feature_helper_sha256"],
    }
    if (any(contract.get(k) != v for k, v in expected.items())
            or final["step"] != 64 or zero["step"] != 0 or zero["contract"] != contract
            or canonical(candidate) != canonical(
                {"schema": "own-search-residual-critic18-candidate-v1",
                 "contract": contract, "params": final["params"]})):
        raise ValueError("exact final64/fresh0/input/source contract differs")
    for path, expected_sha in protocol["training_sources"].items():
        if sha(path) != expected_sha:
            raise ValueError("training source changed")
    base = Path(protocol["critic18"]["path"]).parent
    names = {"critic18": "residual_critic18.py", "features18": "feature_bundle18.py",
             "reanalyze": "reanalyze.py", "trainer18": "train_residual18.py",
             "runner18": "training_cli18_v2.py"}
    if set(reg["closure"]) != set(names):
        raise ValueError("complete qualified training inventory required")
    for key, name in names.items():
        if sha(base / name) != reg["closure"][key]:
            raise ValueError("registered training code changed")
    for key, expected_sha in reg["helper_hashes"].items():
        if sha(Path(reg["helper_dir"]) / (key + ".py")) != expected_sha:
            raise ValueError("registered classical helper changed")
    for path, expected_sha in contract["inference_closure"].items():
        if sha(path) != expected_sha:
            raise ValueError("compiled inference source changed")
    module = load(protocol["critic18"]["path"], protocol["critic18"]["sha256"])
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        for state in [zero, final]:
            restored = module.ResidualCritic18(seed=seed, contract=contract, state=state)
            if canonical(restored.native()) != canonical(state):
                raise ValueError("native full-state roundtrip differs")
        if any(np.count_nonzero(zero["params"][k]) for k in ["w2", "b2"]):
            raise ValueError("fresh additive head was not zero")
        if final["params"] == zero["params"]:
            raise ValueError("final trained parameters are unchanged")
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    fresh = protocol["fresh_final_load_receipt"]
    receipt = json.loads(Path(fresh["path"]).read_bytes())
    if (sha(fresh["path"]) != fresh["sha256"]
            or receipt != fresh["expected_body"]):
        raise ValueError("ROOT actual four fresh-interpreter final loads required")
    if (receipt.get("schema") != "own-search-C18-final-four-fresh-loads-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("source_commit") != protocol["source_commit"]
            or receipt.get("fresh_interpreter_loads") != 4
            or receipt.get("native_sha256_by_seed") != fresh["native_sha256_by_seed"]):
        raise ValueError("four fresh final loads success/bindings required")
    binding = fresh["native_sha256_by_seed"][str(seed)]
    if binding != [row["fresh0_sha256"], row["native_sha256"]]:
        raise ValueError("fresh interpreter receipt native binding differs")
    clock = reg["clocks"]["fit"]
    return {"seed": seed, "status": "strict-C18-frozen-full-native64-PASS",
            "candidate_sha256": row["sha256"], "native_sha256": row["native_sha256"],
            "fresh0_sha256": row["fresh0_sha256"], "step": 64,
            "original_fit_clock": clock, "training_restarted": False}
