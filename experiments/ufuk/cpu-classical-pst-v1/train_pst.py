"""Offline fresh PST fit on exact classical18 completed-game data only."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

from prepare_pst import prepare
from pst_learner import PSTLearner, make_contract
from pst_native import canonical, decode, encode


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_runtime():
    path = Path(__file__).with_name("frozen_reference") / "runtime.py"
    spec = importlib.util.spec_from_file_location("pst_frozen_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_helpers(protocol):
    expected_paths = {
        "pst_features.py": Path(__file__).with_name("pst_features.py"),
        "pst_value.py": Path(__file__).with_name("pst_value.py"),
        "pst_learner.py": Path(__file__).with_name("pst_learner.py"),
        "pst_native.py": Path(__file__).with_name("pst_native.py"),
        "prepare_pst.py": Path(__file__).with_name("prepare_pst.py"),
        "launch_pst.py": Path(__file__).with_name("launch_pst.py"),
        "train_pst.py": Path(__file__),
        "reference_learner.py": Path(__file__).with_name("frozen_reference")
        / "learner.py",
        "reference_journal.py": Path(__file__).with_name("frozen_reference")
        / "journal_v3.py",
        "reference_value.py": Path(__file__).with_name("frozen_reference") / "value.py",
        "reference_runtime.py": Path(__file__).with_name("frozen_reference")
        / "runtime.py",
    }
    registered = protocol.get("helper_sha256", {})
    if set(registered) != set(expected_paths):
        raise ValueError("entire PST helper closure must be registered")
    for name, path in expected_paths.items():
        if sha(path) != registered[name]:
            raise ValueError(f"PST helper closure mismatch: {name}")


def publish(path, payload, *, ceiling=512 * 1024):
    path = Path(path)
    data = payload if isinstance(payload, bytes) else canonical(payload)
    if len(data) > ceiling:
        raise ValueError("PST candidate/native file exceeds 512KiB")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def validate_output_mode(output, *, resume, audit_only):
    """Require an existing branch only for resume/read-only audit."""
    output = Path(output)
    if resume is not None or audit_only:
        if not output.is_dir():
            raise FileNotFoundError("resume/audit output branch must already exist")
    elif output.exists():
        raise FileExistsError(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "config", "journal", "output", "source-repo"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--phase", choices=("proof", "fit"), required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cpu-core", type=int, required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--stop-at", type=int)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--resume-sha256")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--contract-file", type=Path)
    ap.add_argument("--contract-sha256")
    args = ap.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise ValueError("PST protocol SHA differs")
    protocol = json.loads(args.protocol.read_text())
    phase_clock = protocol.get("phase_clocks", {}).get(args.phase, {})
    if (
        protocol.get("schema") != "classical-own-pst-offline-protocol-v1"
        or protocol.get("status") != "registered-pst-proposal-fit-not-strength"
        or args.seed not in protocol["seeds"]
        or args.cpu_core != protocol.get("cpu_core")
        or type(phase_clock.get("first_epoch")) not in (float, int)
        or type(phase_clock.get("deadline_epoch")) not in (float, int)
        or args.deadline != phase_clock.get("deadline_epoch")
        or not phase_clock["first_epoch"] < phase_clock["deadline_epoch"]
        or time.time() < phase_clock["first_epoch"]
        or abs(
            phase_clock["deadline_epoch"] - phase_clock["first_epoch"]
            - (600.0 if args.phase == "proof" else 1800.0)
        ) > 1e-6
    ):
        raise ValueError("registered PST protocol/seed/phase clock required")
    verify_helpers(protocol)
    config_path = Path(protocol["config_paths"][str(args.seed)])
    journal_path = Path(protocol["journal_paths"][str(args.seed)])
    if (
        args.config.resolve() != config_path.resolve()
        or args.journal.resolve() != journal_path.resolve()
    ):
        raise ValueError("only the registered frozen config/journal paths are accepted")
    config = json.loads(config_path.read_text())
    if (
        config["source_commit"] != protocol["source_commit"]
        or config["seed"] != args.seed
    ):
        raise ValueError("actor config source/seed differs")
    if sha(protocol["prior_model_path"]) != protocol["prior_model_sha256"]:
        raise ValueError("immutable human-prior model bytes differ")
    if config["model_sha256"] != protocol["prior_model_sha256"]:
        raise ValueError("actor does not bind the registered human prior")
    runtime = load_frozen_runtime()
    runtime.affinity(args.cpu_core)
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.source_repo, text=True
        ).strip()
        != protocol["source_commit"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=args.source_repo, text=True
        ).strip()
    ):
        raise ValueError("pinned clean producer source required")

    def guard():
        runtime.guard(args.deadline, args.source_repo, args.output.parent)

    validate_output_mode(args.output, resume=args.resume, audit_only=args.audit_only)
    if args.audit_only:
        if args.resume is None or args.contract_file is None:
            raise ValueError(
                "strict native load requires explicit checkpoint and contract"
            )
        if (
            sha(args.resume) != args.resume_sha256
            or sha(args.contract_file) != args.contract_sha256
        ):
            raise ValueError("strict native checkpoint/contract SHA differs")
        contract = json.loads(args.contract_file.read_text())
        if (
            contract.get("schema") != "classical-own-pst-fit-contract-v1"
            or contract.get("protocol_sha256") != args.protocol_sha256
            or contract.get("journal_sha256")
            != protocol["inputs"][str(args.seed)]["journal_sha256"]
            or contract.get("config_sha256")
            != protocol["inputs"][str(args.seed)]["config_sha256"]
            or contract.get("linear_dataset_sha256")
            != protocol["inputs"][str(args.seed)]["classical18_dataset_sha256"]
            or contract.get("dataset_sha256")
            != protocol["inputs"][str(args.seed)]["PST_dataset_sha256"]
            or contract.get("seed") != args.seed
            or contract.get("source_commit") != protocol["source_commit"]
        ):
            raise ValueError("strict-native fit contract/input binding differs")
        runtime.guard(args.deadline, args.source_repo, args.output.parent)
        state = decode(args.resume.read_bytes(), contract)
        model = PSTLearner(args.seed, contract, state)
        if canonical(model.native()) != canonical(state):
            raise ValueError("fresh strict PST native round trip changed full state")
        runtime.guard(args.deadline, args.source_repo, args.output.parent)
        print(json.dumps({"status": "strict-pst-native-load-PASS", "step": model.step}))
        return

    if time.time() >= args.deadline:
        raise TimeoutError("original PST fit clock already expired")
    train, val, receipt, dataset_sha = prepare(
        journal_path, config, protocol["protected_position_keys"]
    )
    guard()
    contract = make_contract(
        protocol, args.protocol, config, journal_path, args.seed, receipt, dataset_sha
    )
    guard()
    state = None
    if args.resume is not None:
        if sha(args.resume) != args.resume_sha256:
            raise ValueError("PST native SHA differs")
        state = decode(args.resume.read_bytes(), contract)
    learner = PSTLearner(args.seed, contract, state)
    if state is None:
        guard()
        args.output.mkdir(parents=True)
        guard()
        publish(args.output / "fit-contract.json", canonical(contract))
        guard()
        publish(
            args.output / "step-00000000.native.gz", encode(learner.native(), contract)
        )
    else:
        contract_path = args.output / "fit-contract.json"
        if (
            not contract_path.is_file()
            or json.loads(contract_path.read_text()) != contract
        ):
            raise ValueError("existing PST output contract differs")
        original = decode(
            (args.output / "step-00000000.native.gz").read_bytes(), contract
        )
        if original["contract"] != contract:
            raise ValueError("existing PST output belongs to another fit")
    stop = contract["updates"] if args.stop_at is None else args.stop_at
    if type(stop) is not int or not learner.step <= stop <= contract["updates"]:
        raise ValueError("PST fixed stop cursor invalid")
    while learner.step < stop:
        guard()
        next_step = min(stop, learner.step + (4 if stop <= 8 else 128))
        learner.advance(train, next_step, guard)
        guard()
        checkpoint = args.output / f"step-{next_step:08d}.native.gz"
        publish(checkpoint, encode(learner.native(), contract))
    if learner.step == contract["updates"]:
        guard()
        publish(args.output / "candidate.json", canonical(learner.candidate()))
    print(
        json.dumps(
            {
                "status": "completed-pst-fit-not-strength"
                if learner.step == contract["updates"]
                else "pst-native-paused-not-candidate",
                "step": learner.step,
                "total_updates": contract["updates"],
                "training_rows": receipt["training_rows"],
                "validation_rows": receipt["validation_rows"],
                "validation_used_for_selection": False,
            }
        )
    )


if __name__ == "__main__":
    main()
