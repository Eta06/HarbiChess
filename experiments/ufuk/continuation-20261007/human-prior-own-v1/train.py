"""CPU teacher-free own/synthetic phases; strict full native and source contracts."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from native import Learner, bootstrap_weights, load_native


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_once(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)
    if Path(path).stat().st_size > 8 * 2**20:
        raise ValueError("native8MiB exceeded; artifact/failure preserved")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["contract", "dataset", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--bootstrap-candidate", type=Path)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only qualification")
    contract = json.loads(a.contract.read_bytes())
    core = Path(contract["core_source_repo"])
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core, text=True
    ).strip() != contract["core_source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=core, text=True
    ):
        raise ValueError("clean original core producer pin required")
    if sha(a.dataset) != contract["dataset_sha256"]:
        raise ValueError("exact data bytes")
    if set(contract["source_sha256"]) != {
        str(Path(__file__).with_name(name).resolve())
        for name in ["model.py", "native.py", "train.py"]
    }:
        raise ValueError("complete training source closure required")
    for path, expected in contract["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError("pinned training source differs")
    if a.dataset.stat().st_size > 8 * 2**20:
        raise ValueError("dataset8MiB cap")
    data = json.loads(a.dataset.read_bytes())
    if (
        data["schema"] != "own-kingbucket-sparse-training-data-v1"
        or data["phase"] != contract["phase"]
    ):
        raise ValueError("target phase/provenance must be explicit")
    rows = data["rows"]
    if not rows:
        raise ValueError("nonempty frozen dataset")
    if contract["phase"] != "synthetic-test" and (
        not contract.get("target_provenance_sha256") or not contract.get("prior_helper_sha256")
    ):
        raise ValueError("teacher/own target and frozen prior source required")
    if contract["phase"] != "synthetic-test":
        for name in ["target_provenance", "prior_helper"]:
            if sha(contract[name + "_path"]) != contract[name + "_sha256"]:
                raise ValueError("exact target/prior helper input provenance")
        if not contract.get("inference_source_sha256"):
            raise ValueError("complete C/Python inference closure required")
        for path, expected in contract["inference_source_sha256"].items():
            if sha(path) != expected:
                raise ValueError("inference source changed")
    for path, expected in contract["execution_helpers_sha256"].items():
        if sha(path) != expected:
            raise ValueError("versioned execution helper source changed")
    if not contract.get("synthetic_only"):
        from convert import pins_tree

        pins_tree(contract["raw_collection_inputs"])
    first, end = contract["original_first_epoch"], contract["original_deadline_epoch"]
    if (
        contract.get("execution_scope_schema") != "human-prior-own-execution-contract-v1"
        or contract.get("execution_mode") not in ["proof", "fresh-fit"]
        or not first < end <= contract["operator_end_epoch"] <= 1791448916.685839
        or end - first > (600 if contract["execution_mode"] == "proof" else 1800)
    ):
        raise ValueError("operator hard cap and separate prospective phase clock")
    import shutil
    import sys

    sys.path.insert(0, contract["core_source_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256MiB")
        if not first <= time.time() < end:
            raise TimeoutError("original registered phase clock; no reset")

    guard()
    if a.resume:
        if a.bootstrap_candidate or sha(a.resume) != a.resume_sha256:
            raise ValueError("strict native resume cannot also take weights bridge")
        learner = load_native(a.resume, contract)
    else:
        weights = None
        if a.bootstrap_candidate:
            if sha(a.bootstrap_candidate) != contract["bootstrap_candidate_sha256"]:
                raise ValueError("named bootstrap candidate SHA")
            if str(a.bootstrap_candidate.resolve()) != contract["bootstrap_candidate_path"]:
                raise ValueError("exact weights-only bootstrap input path")
            weights = bootstrap_weights(contract)
        learner = Learner(contract, weights_initializer=weights)
    if a.audit_only:
        guard()
        print(json.dumps({"status": "PASS-strict-native-readonly", "step": learner.step}))
        return
    a.output.mkdir(exist_ok=False)
    save_once(a.output / "initial.pt", learner.native())
    learner.advance(rows, a.stop, guard)
    guard()
    save_once(a.output / "native.pt", learner.native())
    if learner.step == contract["updates"]:
        save_once(a.output / "candidate.pt", learner.candidate())
    # Atomic/publish-once, no alleged portable torch ZIP byte identity.
    result = dict(
        status="PASS-fixed-stop-not-strength",
        step=learner.step,
        native_sha256=sha(a.output / "native.pt"),
        finished=time.time(),
    )
    with (a.output / "result.json").open("x") as f:
        json.dump(result, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
