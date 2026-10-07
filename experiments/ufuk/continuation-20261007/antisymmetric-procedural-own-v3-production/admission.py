"""Full original phase/source/data/native/Adam/RNG six+two admission, no fabricated claim."""

import json
from pathlib import Path

import torch
from native import bits_equal, load_native
from support import canonical, pinned, pins_tree, read, sha
from zero_parent import admit


def phase(result_ref, contract_ref, mode):
    c = read(contract_ref)
    r = read(result_ref)
    pins_tree(c)
    if c.get("synthetic_only"):
        raise ValueError("synthetic source fixtures cannot qualify actual phase")
    expected = [0, 8, 0, 4, 4, 8] if mode == "proof" else [0, 64]
    if (
        r["schema"] != "antisymmetric-full-native-phase-result-v3"
        or r["status"] != "PASS-antisymmetric-fixed-phase-and-fresh-native-loads-not-strength"
        or r["mode"] != mode
        or c["mode"] != mode
        or r["contract_sha256"] != contract_ref["sha256"]
        or r["weights_only_initializer"] != c["zero_parent"]["candidate"]
        or [x["step"] for x in r["native_payloads"]] != expected
        or not c["first"] == r["first"] < r["finished"] < r["deadline"] == c["deadline"]
        or (mode == "proof" and r["full_payload_bits_equal"] is not True)
    ):
        raise ValueError("actual phase clocks/exact namedzero/native inventory")
    states = []
    for item in r["native_payloads"]:
        pinned(item)
        log = Path(item["log_path"])
        if json.loads(log.read_bytes()) != dict(
            status="PASS-strict-antisymmetric-native-readonly", step=item["step"]
        ):
            raise ValueError("every actual fresh strict native result")
        matching = [
            cmd
            for cmd in r["commands"]
            if "--audit-only" in cmd["command"]
            and cmd["command"][cmd["command"].index("--resume") + 1] == item["path"]
        ]
        if len(matching) != 1:
            raise ValueError("one fresh interpreter command per native")
        cmd = matching[0]
        argv = cmd["command"]
        if (
            cmd["returncode"] != 0
            or not c["first"] <= cmd["started"] <= cmd["finished"] <= c["deadline"]
            or sha(log) != cmd["log_sha256"]
            or argv[1] != c["source_refs"]["train.py"]["path"]
            or argv[argv.index("--contract") + 1] != contract_ref["path"]
        ):
            raise ValueError("source/native/clock/fresh interpreter/log binding")
        state = load_native(pinned(item), c).native()
        if state["step"] != item["step"]:
            raise ValueError("native/Adam counter")
        states.append(state)
    if mode == "proof" and any(
        not bits_equal(states[a], states[b]) for a, b in [(0, 2), (1, 5), (3, 4)]
    ):
        raise ValueError("allstate storage/RNG replay mismatch")
    return c, states, r


def child(binding):
    c, states, r = phase(binding["fit_result"], binding["fit_contract"], "fresh-fit")
    pc, proof, _ = phase(binding["proof_result"], binding["proof_contract"], "proof")
    allowed = {
        "first",
        "deadline",
        "original_first_epoch",
        "original_deadline_epoch",
        "mode",
        "own_proof_result",
        "own_proof_contract",
    }
    if {k: v for k, v in c.items() if k not in allowed} != {
        k: v for k, v in pc.items() if k not in allowed
    }:
        raise ValueError("same data/parent/math/source; only new phase clock and proof refs")
    admit(c["zero_parent"], c["seed"])
    from contracts import build

    for ref in [binding["proof_build_seal"], binding["fit_build_seal"]]:
        built = build(read(ref))
        expected = pc if built["mode"] == "proof" else c
        if canonical(built) != canonical(expected):
            raise ValueError("full original common-data contract replay")
    learner = load_native(pinned(r["native_payloads"][1]), c)
    packet = torch.load(pinned(binding["candidate"]), weights_only=False, map_location="cpu")
    if not bits_equal(packet, learner.candidate()) or bits_equal(
        states[0]["model"], states[1]["model"]
    ):
        raise ValueError("actual nonzero trained candidate/final64 fullstate")
    return packet, c
