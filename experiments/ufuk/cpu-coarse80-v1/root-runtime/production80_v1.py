"""ROOT registered real-data qualification, fresh fit, and strict fresh native loads.

Phase-specific closure binds original clock and all executed source/data bytes.
Proof models are never used to initialize a production fit.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def guard(reg):
    assert reg["first_epoch"] <= time.time() < reg["deadline_epoch"] <= 1791273600
    for path, digest in reg["input_pins"].items():
        assert sha(path) == digest, path


def prepare(reg, seed, value):
    from dataset80 import labels_to_groups, replay_legal_position

    path = reg["labels"][str(seed)]
    payload = json.loads(gzip.decompress(Path(path).read_bytes()))
    assert payload["seed"] == seed and payload["source_commit"] == reg["core_commit"]
    assert payload["frozen_prior_model"] == value.model_dict()
    assert payload["search"] == reg["label_search"]
    assert payload["registration_sha256"] == reg["label_registrations"][str(seed)]
    groups = labels_to_groups(
        payload,
        replay_position=replay_legal_position,
        classical_features=value.features,
        prior_weights=value.PRIOR,
        prior_scale=value.SCALE,
    )
    assert len(groups) == reg["expected_groups"][str(seed)]
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=["proof", "fit", "resume", "load"], required=True
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--target", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    os.sched_setaffinity(0, {reg["cpu_core"]})
    guard(reg)
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=reg["checkout"], text=True
        ).strip()
        == reg["core_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=reg["checkout"], text=True
    )
    sys.path.insert(0, reg["coarse_directory"])
    from learner80 import Learner, decode_native, encode_native, canonical, OPTIMIZER
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "coarse80_actual_prior", reg["classical_value"]
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    if args.mode in ["resume", "load"]:
        assert (
            args.seed in reg["seeds"]
            and args.native is not None
            and args.out is not None
        )
        contract = reg["contracts"][str(args.seed)]
        blob = args.native.read_bytes()
        state = decode_native(blob, contract)
        model = Learner(args.seed, contract, state=state)
        if args.mode == "resume":
            groups = prepare(reg, args.seed, value)
            model.advance(groups, args.target, guard=lambda: guard(reg))
            args.out.write_bytes(encode_native(model.native()))
        else:
            assert encode_native(model.native()) == blob
            assert model.step == args.target
            write(
                args.out,
                dict(
                    status="PASS-strict-fresh-native-full-state",
                    seed=args.seed,
                    step=model.step,
                    native_sha256=sha(args.native),
                    global_rng_equal=model.native()["global_rng"] == random.getstate(),
                    contract=contract,
                    optimizer_steps=0,
                    finished_epoch=time.time(),
                ),
            )
        guard(reg)
        return
    assert args.mode == reg["phase"] and reg["schema"] == "ownq-coarse80-root-phase-v2"
    preflight = json.loads(Path(reg["architecture_receipt"]).read_text())
    assert preflight["status"] == "PASS" and preflight["latency"]["passes_1_10_gate"]
    if args.mode == "fit":
        proof = json.loads(Path(reg["proof_receipt"]).read_text())
        assert proof["status"] == "PASS-real1024-native-qualification-both-seeds"
        assert proof["finished_epoch"] <= reg["first_epoch"]
    output = Path(reg["output"])
    output.mkdir(exist_ok=False)
    rows = []
    for seed in reg["seeds"]:
        guard(reg)
        groups = prepare(reg, seed, value)
        contract = reg["contracts"][str(seed)]
        assert (
            contract["optimizer"] == OPTIMIZER
            and contract["closure_sha256"]
            == hashlib.sha256(canonical(reg["contract_closure"])).hexdigest()
        )
        root = output / str(seed)
        root.mkdir()
        model = Learner(seed, contract)
        (root / "fresh0.native.gz").write_bytes(encode_native(model.native()))
        if args.mode == "proof":
            for step in [4, 8]:
                model.advance(groups, step, guard=lambda: guard(reg))
                (root / f"whole{step}.native.gz").write_bytes(
                    encode_native(model.native())
                )
            split = Learner(seed, contract)
            split.advance(groups, 4, guard=lambda: guard(reg))
            (root / "pause4.native.gz").write_bytes(encode_native(split.native()))
            assert (root / "pause4.native.gz").read_bytes() == (
                root / "whole4.native.gz"
            ).read_bytes()
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--registration",
                    str(args.registration),
                    "--mode",
                    "resume",
                    "--seed",
                    str(seed),
                    "--native",
                    str(root / "pause4.native.gz"),
                    "--target",
                    "8",
                    "--out",
                    str(root / "resumed8.native.gz"),
                ],
                check=True,
                timeout=max(1, reg["deadline_epoch"] - time.time()),
            )
            assert (root / "resumed8.native.gz").read_bytes() == (
                root / "whole8.native.gz"
            ).read_bytes()
            load_specs = [
                (root / "fresh0.native.gz", 0),
                (root / "whole4.native.gz", 4),
                (root / "whole8.native.gz", 8),
            ] * 2
        else:
            model.advance(groups, 64, guard=lambda: guard(reg))
            (root / "final64.native.gz").write_bytes(encode_native(model.native()))
            assert model.step == 64
            write(
                root / "candidate.json",
                dict(
                    schema="coarse80-real-ownq-candidate-v1",
                    model=model.native()["candidate"],
                    contract=contract,
                    registration_sha256=sha(args.registration),
                    native_sha256=sha(root / "final64.native.gz"),
                    seed=seed,
                ),
            )
            load_specs = [
                (root / "fresh0.native.gz", 0),
                (root / "final64.native.gz", 64),
            ]
        for index, (native, step) in enumerate(load_specs):
            guard(reg)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--registration",
                    str(args.registration),
                    "--mode",
                    "load",
                    "--seed",
                    str(seed),
                    "--native",
                    str(native),
                    "--target",
                    str(step),
                    "--out",
                    str(root / f"fresh-load-{index}.json"),
                ],
                check=True,
                timeout=max(1, reg["deadline_epoch"] - time.time()),
            )
        rows.append(
            dict(
                seed=seed,
                groups=len(groups),
                rows=sum(map(len, groups.values())),
                fixed_updates=model.step,
                fresh_loads=len(load_specs),
                qualified_model_used_as_initializer=False,
            )
        )
    guard(reg)
    write(
        output / "result.json",
        dict(
            schema="ownq-coarse80-realdata-phase-result-v1",
            status="PASS-real1024-native-qualification-both-seeds"
            if args.mode == "proof"
            else "PASS-FRESH64-ownQ-fits-and-four-freshloads",
            phase=args.mode,
            rows=rows,
            first_epoch=reg["first_epoch"],
            deadline_epoch=reg["deadline_epoch"],
            finished_epoch=time.time(),
            registration_sha256=sha(args.registration),
            strength_success_claimed=False,
        ),
    )


if __name__ == "__main__":
    main()
