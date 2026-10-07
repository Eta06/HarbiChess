"""ROOT own-phase proof or fresh64; named teacher WEIGHTS only, full state strict loads."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from contract_builder import canonical, pinned, sha
from native import PHASE, SCHEMA

REGISTRATION_SCHEMA = "NNUE-own-closed-terminal-training-orchestration-v1"
PROOF_STATUS = "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength"


def execute(reg):
    if reg["schema"] != REGISTRATION_SCHEMA or reg["status"] != "registered":
        raise ValueError("ROOT closed-terminal own-WDL-specific immutable registration")
    first, end = reg["first"], reg["deadline"]
    proof = reg["mode"] == "proof"
    if (
        reg["mode"] not in ["proof", "fresh-fit"]
        or not first <= time.time() < end <= reg["operator_end_epoch"] <= 1791448916.685839
        or end - first > (600 if proof else 1800)
    ):
        raise ValueError("separate original600/1800 clock")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    train = pinned(reg["train"])
    sys.path.insert(0, str(train.parent))
    native_path = pinned(reg["native"])
    if sha(native_path) != reg["native"]["sha256"]:
        raise ValueError("strict native source binding")
    sys.path.insert(0, str(native_path.parent))
    import native as native

    cp, data = pinned(reg["contract"]), pinned(reg["dataset"])
    provenance = pinned(reg["target_provenance"])
    parent = pinned(reg["parent_candidate"])
    c = json.loads(cp.read_bytes())
    if (
        c["phase"] != PHASE
        or c["native_schema"] != SCHEMA
        or c["updates"] != 64
        or c["operator_end_epoch"] != reg["operator_end_epoch"]
        or c["execution_mode"] != reg["mode"]
        or c["math"] != native.MATH
        or c["seed"] != reg["seed"]
        or c["dataset_sha256"] != sha(data)
        or c["target_provenance_sha256"] != sha(provenance)
        or c["original_first_epoch"] != first
        or c["original_deadline_epoch"] != end
        or c["bootstrap_candidate_sha256"] != sha(parent)
        or c["bootstrap_candidate_path"] != str(parent.resolve())
        or c["parent_candidate_sha256"] != sha(parent)
    ):
        raise ValueError("strict closed-terminal own-WDL64/same-parent weights-only contract")
    for row in reg["inputs"].values():
        pinned(row)
    for path, h in reg["source_sha256"].items():
        pinned(dict(path=path, sha256=h))
    sys.path.insert(0, c["core_source_repo"] + "/src")
    import shutil

    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    out = Path(reg["output"])
    if not out.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM output")
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def guard():
        budget.check()
        if not first <= time.time() < end:
            raise TimeoutError("original phase clock including all audit work")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("disk floor256")
        if sum(x.stat().st_size for x in out.rglob("*") if x.is_file()) > 96 * 2**20:
            raise RuntimeError("phase RAM96MiB")

    def run(name, stop, resume=None, audit=False):
        guard()
        cmd = [
            sys.executable,
            str(train),
            "--contract",
            str(cp),
            "--dataset",
            str(data),
            "--output",
            str(out / name),
            "--stop",
            str(stop),
        ]
        if resume is not None:
            cmd += ["--resume", str(resume), "--resume-sha256", sha(resume)]
        else:
            cmd += ["--bootstrap-candidate", c["bootstrap_candidate_path"]]
        if audit:
            cmd += ["--audit-only"]
        with (out / (name + ".log")).open("xb") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            owner = dict(
                command=cmd,
                pid=proc.pid,
                startticks=Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
            )
            try:
                while proc.poll() is None:
                    guard()
                    time.sleep(0.05)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
            owner.update(
                returncode=proc.returncode,
                finished=time.time(),
                log_sha256=sha(out / (name + ".log")),
            )
            commands.append(owner)
            if proc.returncode:
                raise RuntimeError("owned child failed " + name)
        guard()
        return out / name

    try:
        stop = 8 if proof else 64
        run("whole", stop)
        paths = [out / "whole/initial.pt", out / "whole/native.pt"]
        if proof:
            run("pause", 4)
            run("resume", 8, out / "pause/native.pt")
            if not native.bits_equal(
                native.load_native(paths[1], c).native(),
                native.load_native(out / "resume/native.pt", c).native(),
            ):
                raise ValueError(
                    "full model/baseline/Adam/globalTorch/privateSampler replay mismatch"
                )
            paths += [
                out / "pause/initial.pt",
                out / "pause/native.pt",
                out / "resume/initial.pt",
                out / "resume/native.pt",
            ]
        checks = []
        for index, path in enumerate(paths):
            guard()
            state = native.load_native(path, c).native()
            step = state["step"]
            run("audit-" + str(index), step, path, True)
            checks.append(
                dict(
                    path=str(path),
                    sha256=sha(path),
                    step=step,
                    actual_fresh_process=True,
                    log_path=str(out / ("audit-" + str(index) + ".log")),
                )
            )
        zero = native.load_native(paths[0], c).native()
        final = native.load_native(paths[1], c).native()
        if (
            zero["step"] != 0
            or zero["optimizer"]["state"]
            or not native.bits_equal(zero["model"], zero["baseline"])
        ):
            raise ValueError("NEW ownAdam and teacherweights-only baseline")
        if not proof:
            if native.bits_equal(zero["model"], final["model"]):
                raise ValueError("own64 must actually change weights")
            import torch

            packet = torch.load(out / "whole/candidate.pt", map_location="cpu", weights_only=False)
            if not native.bits_equal(packet, native.load_native(paths[1], c).candidate()):
                raise ValueError("fixed64 candidate vs fullnative")
        guard()
        result = dict(
            status=PROOF_STATUS,
            mode=reg["mode"],
            own_updates=stop,
            full_payload_bits_equal=proof,
            raw_zip_identity_claimed=False,
            native_payloads=checks,
            contract_sha256=sha(cp),
            registration_sha256=reg["registration_sha256"],
            weights_only_initializer=reg["parent_candidate"],
            teacher_labels_used=False,
            dataset_sha256=sha(data),
            target_provenance_sha256=sha(provenance),
            seed=reg["seed"],
        )
    except BaseException as e:
        result = dict(status="FAILED-preserved", error=repr(e))
        raise
    finally:
        result.update(first=first, deadline=end, finished=time.time(), commands=commands)
        with (out / "result.json").open("xb") as stream:
            stream.write(canonical(result) + b"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", required=True, type=Path)
    path = p.parse_args().registration
    reg = json.loads(path.read_bytes())
    reg["registration_sha256"] = sha(path)
    execute(reg)
