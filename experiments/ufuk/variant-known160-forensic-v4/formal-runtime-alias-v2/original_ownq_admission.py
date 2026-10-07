"""New own64 evidence admission; old teacher clocks remain historical metadata."""

import json
import random
import sys
from pathlib import Path

from teacher_admission import module, pin, read, sha, teacher

SEEDS = (20262905, 20262906)
SEARCH_SHA = "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"


def import_nnue(protocol):
    """Explicit pinned origins; never resolve local profile.py through cProfile."""
    directory = Path(protocol["nnue_directory"]).resolve()
    deps = protocol["nnue_helpers"]
    sys.path.insert(0, str(directory))
    try:
        for name in ["model", "native"]:
            cached = sys.modules.get(name)
            expected = directory / (name + ".py")
            if cached is not None and Path(cached.__file__).resolve() != expected:
                raise ValueError("cached NNUE dependency collision " + name)
            module(deps[name + ".py"], name)
        evaluator = module(deps["evaluator.py"], "nnue_arena_evaluator")
        compiled = module(protocol["binary"], "_kingbucket16")
    finally:
        sys.path.pop(0)
    return sys.modules["model"], sys.modules["native"], evaluator, compiled


def own_receipt(result, contract_ref, contract, steps, operator_end):
    status = result["status"]
    phase_cap = 600 if steps == [0, 8, 0, 4, 4, 8] else 1800
    if result["deadline"] - result["first"] > phase_cap:
        raise ValueError("original600proof/1800fit ceiling")
    if (
        status
        not in [
            "PASS-readonly-loads-and-fixed-phase-not-strength",
            "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength",
        ]
        or result["contract_sha256"] != contract_ref["sha256"]
        or result.get("raw_zip_identity_claimed") is not False
        or not result["first"] <= result["finished"] <= result["deadline"] <= operator_end
        or (result["first"], result["deadline"])
        != (contract["original_first_epoch"], contract["original_deadline_epoch"])
    ):
        raise ValueError("actual successful original own phase clock/contract")
    audits = [r for r in result["commands"] if "--audit-only" in r["command"]]
    if len(audits) != len(steps) or any(
        r.get("code", r.get("returncode")) != 0 for r in result["commands"]
    ):
        raise ValueError("all actual fresh child processes required")
    refs = []
    for command, step in zip(audits, steps, strict=True):
        argv = command["command"]
        path = argv[argv.index("--resume") + 1]
        h = argv[argv.index("--resume-sha256") + 1]
        if (
            argv[argv.index("--contract") + 1] != contract_ref["path"]
            or int(argv[argv.index("--stop") + 1]) != step
        ):
            raise ValueError("fresh child exact contract/counter")
        if "stdout" in command:
            stdout = command["stdout"]
            stderr = command["stderr"]
        else:
            payload = next(x for x in result["native_payloads"] if x["path"] == path)
            if payload["step"] != step or payload["actual_fresh_process"] is not True:
                raise ValueError("actual native fresh-process record")
            log = pin(dict(path=payload["log_path"], sha256=command["log_sha256"]))
            stdout = log.read_text()
            stderr = ""
        if json.loads(stdout) != dict(status="PASS-strict-native-readonly", step=step) or stderr:
            raise ValueError("actual child strict payload load PASS output")
        refs.append(dict(path=path, sha256=h, step=step))
        pin(refs[-1])
    return refs


def proof_fit_contracts(proof, fit, proof_result, proof_contract_ref=None):
    ignored = {
        "original_first_epoch",
        "original_deadline_epoch",
        "inference_source_sha256",
        "execution_mode",
        "own_phase_proof",
        "contract_build_seal",
    }
    if {k: v for k, v in proof.items() if k not in ignored} != {
        k: v for k, v in fit.items() if k not in ignored
    }:
        raise ValueError("ALL own proof/fit data/source/math/initializer bindings")
    if (
        proof["execution_mode"] != "proof"
        or fit["execution_mode"] != "fresh-fit"
        or proof["own_phase_proof"] is not None
        or fit["own_phase_proof"] != proof_result
    ):
        raise ValueError("explicit proof-to-fresh64 metadata transition")
    before, after = proof["inference_source_sha256"], fit["inference_source_sha256"]
    if (
        any(after.get(k) != v for k, v in before.items())
        or set(after) - set(before) != {proof_result["path"]}
        or after[proof_result["path"]] != proof_result["sha256"]
    ):
        raise ValueError("fresh fit exact own proof receipt")
    a, b = proof["contract_build_seal"], fit["contract_build_seal"]
    phase_keys = {"first", "deadline", "mode", "own_proof_result", "own_proof_contract"}
    if (
        {k: v for k, v in a.items() if k not in phase_keys}
        != {k: v for k, v in b.items() if k not in phase_keys}
        or a["mode"] != "proof"
        or b["mode"] != "fresh-fit"
        or b["own_proof_result"] != proof_result
        or b["own_proof_contract"] != proof_contract_ref
    ):
        raise ValueError("exact v2 build seal inputs; ONLY clock/mode/namedproof differs")
    for contract, seal in [(proof, a), (fit, b)]:
        if (contract["original_first_epoch"], contract["original_deadline_epoch"]) != (
            seal["first"],
            seal["deadline"],
        ):
            raise ValueError("actual build seal clock matches native contract")


def child(protocol, seed, native):
    if seed not in SEEDS:
        raise ValueError("fixed seeds")
    parent_info = protocol["teachers"][str(seed)]
    info = protocol["children"][str(seed)]
    import torch

    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    try:
        parent, _, parent_result = teacher(parent_info, native)
    finally:
        random.setstate(before_python)
        torch.set_rng_state(before_torch)
    proof, c = read(info["proof_contract"]), read(info["contract"])
    if (
        c["phase"] != "own-learning"
        or c["updates"] != 64
        or c["math"] != native.MATH
        or c["seed"] != seed
        or c["teacher_ancestry"] != parent_info
        or c["bootstrap_candidate_path"] != parent_info["candidate"]["path"]
        or c["bootstrap_candidate_sha256"] != parent_info["candidate"]["sha256"]
    ):
        raise ValueError("fresh own64 exact teacher WEIGHTS-only initializer")
    proof_fit_contracts(proof, c, info["proof_result"], info["proof_contract"])
    for mode, key in [("proof", "proof_registration"), ("fresh-fit", "fit_registration")]:
        registration = read(info[key])
        result = read(info["proof_result" if mode == "proof" else "fit_result"])
        if (
            registration["schema"] != "NNUE-own-training-orchestration-v2"
            or registration["mode"] != mode
            or registration["seed"] != seed
            or result["registration_sha256"] != info[key]["sha256"]
        ):
            raise ValueError("actual ROOT own v2 producer registration")
    for contract in [proof, c]:
        for field in ["source_sha256", "inference_source_sha256"]:
            for path, h in contract[field].items():
                pin(dict(path=path, sha256=h))
        for field in ["prior_helper", "target_provenance"]:
            pin(dict(path=contract[field + "_path"], sha256=contract[field + "_sha256"]))
    data = read(info["dataset"])
    if (
        sha(pin(info["dataset"])) != c["dataset_sha256"]
        or data["phase"] != "own-learning"
        or data["schema"] != "own-kingbucket-sparse-training-data-v1"
        or len(data["rows"]) != 1024
    ):
        raise ValueError("exact NEW own1024 data")
    collection = read(info["collection_receipt"])
    collection_reg = read(info["collection_registration"])
    if (
        collection["schema"] != "own-nnue-ownq-collection-receipt-v2"
        or collection["status"] != "PASS-exact-row-budget"
        or collection["train_rows"] != 1024
        or collection["teacher_labels_used"] is not False
        or collection_reg["seed"] != seed
        or collection["registration_sha256"] != info["collection_registration"]["sha256"]
        or collection["parent_candidate_sha256"] != parent_info["candidate"]["sha256"]
        or not collection["original_first_epoch"]
        <= collection["finished_epoch"]
        <= collection["original_deadline_epoch"]
        <= protocol["ROOToperator_end_epoch"]
    ):
        raise ValueError("actual closed own current-parent data lineage")
    provenance = json.loads(Path(c["target_provenance_path"]).read_bytes())
    if (
        provenance["collection_receipt_sha256"] != info["collection_receipt"]["sha256"]
        or provenance["inputs"] != c["raw_collection_inputs"]
        or provenance["inputs"]["registration"] != info["collection_registration"]
        or provenance["inputs"]["events"] != info["events"]
    ):
        raise ValueError("native raw collection/converted provenance exact binding")
    converter_ref = info["converter"]
    if c["execution_helpers_sha256"].get(converter_ref["path"]) != converter_ref["sha256"]:
        raise ValueError("actual producer-qualified converter source, not new guessed validator")
    converter = module(converter_ref, "nnue_strength_raw_own_converter")
    replayed, trace_bytes = converter.convert(provenance["inputs"])
    if (
        replayed != pin(info["dataset"]).read_bytes()
        or trace_bytes != Path(c["target_provenance_path"]).read_bytes()
    ):
        raise ValueError(
            "independent fullhistory/legal/terminal/protected/alias-span replay byte mismatch"
        )
    checks = []
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    try:
        for contract, ref, steps, mode in [
            (proof, info["proof_result"], [0, 8, 0, 4, 4, 8], "proof"),
            (c, info["fit_result"], [0, 64], "fresh-fit"),
        ]:
            result = read(ref)
            if result["mode"] != mode or (
                mode == "proof" and result["full_payload_bits_equal"] is not True
            ):
                raise ValueError("actual phase scope")
            rows = own_receipt(
                result,
                info["proof_contract"] if mode == "proof" else info["contract"],
                contract,
                steps,
                protocol["ROOToperator_end_epoch"],
            )
            states = []
            for row in rows:
                loaded = native.load_native(pin(row), contract)
                if loaded.step != row["step"]:
                    raise ValueError("native internal counter")
                states.append(loaded.native())
            if mode == "proof":
                if (
                    not native.bits_equal(states[1], states[5])
                    or not native.bits_equal(states[0], states[2])
                    or not native.bits_equal(states[3], states[4])
                ):
                    raise ValueError("independent full model/baseline/Adam/ALL RNG chain differs")
            else:
                if (
                    rows[0]["path"] != info["initial"]["path"]
                    or rows[0]["sha256"] != info["initial"]["sha256"]
                    or rows[1]["path"] != info["native"]["path"]
                    or rows[1]["sha256"] != info["native"]["sha256"]
                ):
                    raise ValueError("actual freshfit0/final64 audited files exact")
            checks.extend(rows)
        zero = native.load_native(pin(info["initial"]), c).native()
        final = native.load_native(pin(info["native"]), c).native()
        candidate = torch.load(pin(info["candidate"]), map_location="cpu", weights_only=False)
        if (
            zero["step"] != 0
            or final["step"] != 64
            or zero["optimizer"]["state"]
            or not native.bits_equal(zero["model"], parent)
            or not native.bits_equal(zero["baseline"], parent)
            or not native.bits_equal(final["baseline"], parent)
            or candidate.keys() != {"schema", "contract", "model"}
            or candidate["contract"] != c
            or candidate["schema"] != "own-kingbucket-nnue16-model-v1"
            or not native.bits_equal(candidate["model"], final["model"])
            or native.bits_equal(candidate["model"], parent)
        ):
            raise ValueError("fresh ownAdam/initialized baseline/fixed changed64/candidate")
        return (
            candidate["model"],
            parent,
            dict(
                seed=seed,
                teacher=parent_result,
                own_updates=64,
                own_proof_fresh_loads=6,
                own_fit_fresh_loads=2,
                full_native_checks=len(checks),
                own_candidate_sha256=info["candidate"]["sha256"],
                teacher_ancestry=True,
            ),
        )
    finally:
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
