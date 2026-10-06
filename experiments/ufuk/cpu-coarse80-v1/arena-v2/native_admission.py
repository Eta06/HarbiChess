"""Strict full native0/final64 plus original proof/fit phase bindings; no SGD."""

import hashlib
import json
import random
import sys
from pathlib import Path

from value import check_sources, sha


def canonical(x):
    return json.dumps(
        x, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def check_phase(reg, result, phase, expected_status, digest):
    if (
        reg["schema"] != "ownq-coarse80-root-phase-v2"
        or reg["phase"] != phase
        or result["schema"] != "ownq-coarse80-realdata-phase-result-v1"
        or result["phase"] != phase
        or result["status"] != expected_status
        or result["registration_sha256"] != digest
        or result["first_epoch"] != reg["first_epoch"]
        or not reg["first_epoch"]
        <= result["finished_epoch"]
        <= result["deadline_epoch"]
        == reg["deadline_epoch"]
        or reg["contract_closure"]["native_contract_version"]
        != "ROOT-phase-bound-v2-not-old-validator-v1"
        or reg["contract_closure"]["phase"] != phase
        or reg["contract_closure"]["first_epoch"] != reg["first_epoch"]
        or reg["contract_closure"]["deadline_epoch"] != reg["deadline_epoch"]
        or reg["contract_closure"]["pins"] != reg["input_pins"]
        or result["strength_success_claimed"]
    ):
        raise ValueError("actual completed same-phase immutable receipt required")
    for path, expected in reg["input_pins"].items():
        if sha(path) != expected:
            raise ValueError("actual phase input changed")
    if {r["seed"] for r in result["rows"]} != set(reg["seeds"]) or len(
        result["rows"]
    ) != 2:
        raise ValueError("exact both seeds required")


def admit(protocol, seed):
    check_sources(protocol)
    binding = protocol["native_phase_bindings"]
    phases = {}
    for phase, status in [
        ("proof", "PASS-real1024-native-qualification-both-seeds"),
        ("fit", "PASS-FRESH64-ownQ-fits-and-four-freshloads"),
    ]:
        b = binding[phase]
        for key in ["registration", "result"]:
            if sha(b[key]["path"]) != b[key]["sha256"]:
                raise ValueError("phase receipt SHA changed")
        reg = json.loads(Path(b["registration"]["path"]).read_bytes())
        result = json.loads(Path(b["result"]["path"]).read_bytes())
        check_phase(reg, result, phase, status, b["registration"]["sha256"])
        phases[phase] = reg, result
    reg, result = phases["fit"]
    proof_reg, proof_result = phases["proof"]
    if proof_result["finished_epoch"] > reg["first_epoch"]:
        raise ValueError("qualification must precede production fit")
    directory = Path(protocol["coarse_directory"])
    sys.path.insert(0, str(directory))
    import learner80
    from learner80 import Learner, decode_native, encode_native

    if Path(learner80.__file__).resolve() != directory / "learner80.py":
        raise ValueError("qualified learner import origin differs")

    if proof_reg["core_commit"] != protocol["source_commit"]:
        raise ValueError("qualified source differs")
    proof_row = next(r for r in proof_result["rows"] if r["seed"] == seed)
    if (
        proof_row["fixed_updates"] != 8
        or proof_row["fresh_loads"] != 6
        or proof_row["qualified_model_used_as_initializer"]
    ):
        raise ValueError("actual proof8 plus six fresh native loads required")
    proof_root = Path(proof_reg["output"]) / str(seed)
    proof_contract = proof_reg["contracts"][str(seed)]
    saved_proof_rng = random.getstate()
    try:
        for i, (name, step) in enumerate(
            [("fresh0", 0), ("whole4", 4), ("whole8", 8)] * 2
        ):
            native_path = proof_root / (name + ".native.gz")
            blob = native_path.read_bytes()
            state = decode_native(blob, proof_contract)
            restored = Learner(seed, proof_contract, state=state)
            load = json.loads((proof_root / f"fresh-load-{i}.json").read_bytes())
            if (
                state["step"] != step
                or encode_native(restored.native()) != blob
                or load["status"] != "PASS-strict-fresh-native-full-state"
                or load["native_sha256"] != sha(native_path)
                or load["step"] != step
                or load["seed"] != seed
                or load["contract"] != proof_contract
                or not load["global_rng_equal"]
                or load["optimizer_steps"] != 0
                or load["finished_epoch"] > proof_reg["deadline_epoch"]
            ):
                raise ValueError(
                    "all six original full-native fresh load bindings required"
                )
        if (proof_root / "whole4.native.gz").read_bytes() != (
            proof_root / "pause4.native.gz"
        ).read_bytes():
            raise ValueError("proof pause4 differs from uninterrupted4")
        if (proof_root / "whole8.native.gz").read_bytes() != (
            proof_root / "resumed8.native.gz"
        ).read_bytes():
            raise ValueError("actual freshprocess8 trajectory differs")
    finally:
        random.setstate(saved_proof_rng)

    contract = reg["contracts"][str(seed)]
    if (
        contract["source_commit"] != protocol["source_commit"]
        or contract["seed"] != seed
        or contract["closure_sha256"]
        != hashlib.sha256(canonical(reg["contract_closure"])).hexdigest()
        or contract["labels_sha256"] != reg["input_pins"][reg["labels"][str(seed)]]
    ):
        raise ValueError("full exact source/data/phase closure contract required")
    model = protocol["models"][str(seed)]["learned"]
    saved = random.getstate()
    try:
        states = []
        for key, step in [("fresh0", 0), ("native", 64)]:
            path = Path(model[key + "_path"])
            if sha(path) != model[key + "_sha256"]:
                raise ValueError("exact native SHA changed")
            raw = path.read_bytes()
            state = decode_native(raw, contract)
            restored = Learner(seed, contract, state=state)
            if state["step"] != step or encode_native(restored.native()) != raw:
                raise ValueError(
                    "full native Adam/global+sampler RNG roundtrip differs"
                )
            states.append(state)
        if any(states[0]["theta"]) or states[0]["theta"] == states[1]["theta"]:
            raise ValueError(
                "fresh zero initializer and actually changed final64 required"
            )
        candidate = Path(model["path"])
        wrapper = json.loads(candidate.read_bytes())
        if (
            sha(candidate) != model["sha256"]
            or set(wrapper)
            != {
                "schema",
                "model",
                "contract",
                "registration_sha256",
                "native_sha256",
                "seed",
            }
            or wrapper["schema"] != "coarse80-real-ownq-candidate-v1"
            or wrapper["seed"] != seed
            or wrapper["contract"] != contract
            or wrapper["registration_sha256"]
            != protocol["native_phase_bindings"]["fit"]["registration"]["sha256"]
            or wrapper["native_sha256"] != model["native_sha256"]
            or wrapper["model"] != states[1]["candidate"]
        ):
            raise ValueError("full candidate envelope/native/source/data mismatch")
    finally:
        random.setstate(saved)
    for i, (key, step) in enumerate([("fresh0", 0), ("native", 64)]):
        load_path = Path(reg["output"]) / str(seed) / f"fresh-load-{i}.json"
        load = json.loads(load_path.read_bytes())
        if (
            load["status"] != "PASS-strict-fresh-native-full-state"
            or load["seed"] != seed
            or load["step"] != step
            or load["native_sha256"] != model[key + "_sha256"]
            or load["contract"] != contract
            or not load["global_rng_equal"]
            or load["optimizer_steps"] != 0
            or load["finished_epoch"] > reg["deadline_epoch"]
        ):
            raise ValueError(
                "actual fresh-process initial/final strict load receipt required"
            )
    r = next(r for r in result["rows"] if r["seed"] == seed)
    if (
        r["fixed_updates"] != 64
        or r["fresh_loads"] != 2
        or r["qualified_model_used_as_initializer"]
    ):
        raise ValueError("fresh production64 and two fresh full loads required")
    return dict(
        seed=seed,
        status="strict-Coarse80-full-native0-final64-PASS",
        step=64,
        candidate_sha256=model["sha256"],
        native_sha256=model["native_sha256"],
        fresh0_sha256=model["fresh0_sha256"],
        training_restarted=False,
        original_fit_clock=dict(
            first=reg["first_epoch"], deadline=reg["deadline_epoch"]
        ),
    )
