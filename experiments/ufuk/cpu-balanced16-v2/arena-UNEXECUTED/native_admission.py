"""Strict offline48 native/load/source/data admission without old TRAIN execution."""

import json
import random
import sys
from pathlib import Path

from support import load, sha


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_fit_qualifications(fit, proof_reg, qualified, profile_binding):
    data_keys = {"dataset", "split"}
    if (
        set(proof_reg["inputs"]) != data_keys
        or set(fit["inputs"]) != data_keys | {"native_qualification", "profile_qualification"}
        or any(fit["inputs"][k] != proof_reg["inputs"][k] for k in data_keys)
    ):
        raise ValueError("exact shared dataset/split and all four production bindings required")
    native_binding = fit["inputs"]["native_qualification"]
    if native_binding != {"path": qualified["path"], "sha256": qualified["sha256"]}:
        raise ValueError("fit native qualification must be the admitted actual proof receipt")
    for key in ["native_qualification", "profile_qualification"]:
        row = fit["inputs"][key]
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("production qualification receipt SHA changed")
    profile_row = fit["inputs"]["profile_qualification"]
    if profile_row != {"path": profile_binding["path"], "sha256": profile_binding["sha256"]}:
        raise ValueError("prospectively pinned architecture profile receipt differs")
    if sha(profile_binding["registration_path"]) != profile_binding["registration_sha256"]:
        raise ValueError("architecture profile registration changed")
    reg = json.loads(Path(profile_binding["registration_path"]).read_bytes())
    profile = json.loads(Path(profile_row["path"]).read_bytes())
    if (
        reg["seed"] != fit["seed"]
        or reg["source_commit"] != fit["source_commit"]
        or reg["helper_sha256"] != fit["helper_sha256"]
        or profile.get("status") != "PASS-architecture-only-not-strength"
        or not reg["first_epoch"] <= profile["finished"] <= profile["deadline"]
        or profile["deadline"] != reg["deadline_epoch"]
        or not 0 < profile["median_ratio"] <= 1.10
        or len(profile["packets"]) != 24
        or sorted(r["role"] for r in profile["packets"]) != ["new-zero"] * 12 + ["old-q2"] * 12
    ):
        raise ValueError("exact completed same-source architecture profile qualification required")
    for row in reg["inputs"].values():
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("architecture profile original inputs changed")


def admit(protocol, seed):
    info = protocol["models"][str(seed)]["learned"]
    pair = protocol["fit_inputs"][str(seed)]
    registration = json.loads(Path(pair["path"]).read_bytes())
    if sha(pair["path"]) != pair["sha256"] or registration["seed"] != seed:
        raise ValueError("actual fresh48 registration binding")
    qualified = protocol["proof_inputs"][str(seed)]
    proof_reg = json.loads(Path(qualified["registration_path"]).read_bytes())
    proof = json.loads(Path(qualified["path"]).read_bytes())
    if (
        sha(qualified["path"]) != qualified["sha256"]
        or sha(qualified["registration_path"]) != qualified["registration_sha256"]
        or proof.get("status")
        != "PASS-balanced-v2-own-data8-4-fresh8-six-fullnative-loads-not-strength"
        or proof["registration_sha256"] != qualified["registration_sha256"]
        or proof["finished"] > proof["deadline"]
        or proof_reg["seed"] != seed
        or len(proof["native_payloads"]) != 6
        or any(proof_reg[k] != registration[k] for k in ["helper_sha256", "source_commit"])
    ):
        raise ValueError("same-source/data actual whole8/pause4/fresh8 six native proof required")
    validate_fit_qualifications(
        registration, proof_reg, qualified, protocol["profile_qualification_bindings"][str(seed)]
    )
    for native, step in zip(proof["native_payloads"], [0, 8, 0, 4, 4, 8], strict=True):
        if (
            sha(native["path"]) != native["sha256"]
            or sha(native["roundtrip_path"]) != native["roundtrip_sha256"]
            or json.loads(Path(native["path"]).read_bytes())["step"] != step
            or canonical(json.loads(Path(native["path"]).read_bytes()))
            != canonical(json.loads(Path(native["roundtrip_path"]).read_bytes()))
        ):
            raise ValueError("actual six proof full native bytes/steps changed")
    states = []
    for path, expected in [
        (info["fresh0_path"], info["fresh0_sha256"]),
        (info["native_path"], info["native_sha256"]),
    ]:
        if sha(path) != expected:
            raise ValueError("all-state SHA differs")
        states.append(json.loads(Path(path).read_bytes()))
    zero, final = states
    candidate = json.loads(Path(info["path"]).read_bytes())
    if (
        sha(info["path"]) != info["sha256"]
        or zero["step"] != 0
        or final["step"] != 48
        or zero["contract"] != final["contract"]
        or canonical(candidate) != canonical(final["model"])
    ):
        raise ValueError("actual fresh0/final48/candidate storage differs")
    directory = Path(protocol["balanced_directory"])
    sys.path.insert(0, str(directory))
    try:
        dep = protocol["balanced_helpers"]["learner_v2.py"]
        module = load(dep["path"], dep["sha256"], "balanced_native_contract")
        import transform as transformed

        dataset = json.loads(Path(registration["inputs"]["dataset"]["path"]).read_bytes())
        split = json.loads(Path(registration["inputs"]["split"]["path"]).read_bytes())
        _, record = transformed.transform(dataset, split)
        expected_contract = dict(
            schema="own-balanced-risk-learning-contract-v2",
            seed=seed,
            updates=48,
            math=module.MATH,
            transform=record,
            inputs=registration["inputs"],
            source_commit=protocol["source_commit"],
            helper_sha256=registration["helper_sha256"],
            first_epoch=registration["first_epoch"],
            deadline_epoch=registration["deadline_epoch"],
        )
    finally:
        sys.path.pop(0)
    if final["contract"] != expected_contract:
        raise ValueError("exact actual original48 contract")
    for path, expected in registration["helper_sha256"].items():
        if sha(path) != expected:
            raise ValueError("all qualified source closure differs")
    for row in registration["inputs"].values():
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("dataset/split changed")
    result = json.loads(Path(info["fit_result_path"]).read_bytes())
    if (
        sha(info["fit_result_path"]) != info["fit_result_sha256"]
        or result.get("status") != "PASS-fixed-stop-not-strength"
        or result["step"] != 48
        or result["native_sha256"] != info["native_sha256"]
        or result["deadline"] != registration["deadline_epoch"]
        or result["finished"] > result["deadline"]
    ):
        raise ValueError("original48 completed within fixed clock")
    directory = Path(protocol["balanced_directory"])
    dep = protocol["balanced_helpers"]["learner_v2.py"]
    rng = random.getstate()
    sys.path.insert(0, str(directory))
    try:
        module = load(dep["path"], dep["sha256"], "selective_native_learner")
        if Path(module.model_dict.__code__.co_filename).resolve() != directory / "transform.py":
            raise ValueError("native model import origin")
        for state in states:
            restored = module.Learner(seed, expected_contract, state)
            if canonical(restored.native()) != canonical(state):
                raise ValueError("model/Adam/RNG storage roundtrip differs")
        if zero["weights"] != [0.0] * 16 or final["weights"] == zero["weights"]:
            raise ValueError("fresh zero and actual trained head required")
    finally:
        random.setstate(rng)
        sys.path.pop(0)
    fresh = protocol["fit_fresh_loads"][str(seed)]
    if sha(fresh["path"]) != fresh["sha256"]:
        raise ValueError("fresh final load receipt changed")
    receipt = json.loads(Path(fresh["path"]).read_bytes())
    if (
        receipt["schema"] != "balanced-v2-fit-fresh-loads-v1"
        or receipt["registration_sha256"] != pair["sha256"]
        or receipt["finished"] > registration["deadline_epoch"]
        or len(receipt["payloads"]) != 2
    ):
        raise ValueError("both actual fresh-process0/final48 loads required")
    for expected, row in zip(states, receipt["payloads"], strict=True):
        if (
            sha(row["path"]) != row["sha256"]
            or json.loads(Path(row["path"]).read_bytes()) != expected
            or sha(row["roundtrip_path"]) != row["roundtrip_sha256"]
            or canonical(json.loads(Path(row["roundtrip_path"]).read_bytes()))
            != canonical(expected)
        ):
            raise ValueError("actual fresh-process full payload differs")
    return dict(
        seed=seed,
        status="strict-selective-Q-full-native48-PASS",
        step=48,
        original_training_deadline=registration["deadline_epoch"],
        training_restarted=False,
    )
