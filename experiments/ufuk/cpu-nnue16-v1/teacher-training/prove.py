"""ROOT-clocked real teacher proof/fit; execute only after registration."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from convert import module, pinned, sha


def execute(reg):
    if reg["schema"] != "NNUE-teacher-training-orchestration-v1" or reg["status"] != "registered":
        raise ValueError("ROOT registration")
    first, end = reg["first"], reg["deadline"]
    proof = reg["mode"] == "proof"
    if reg["mode"] not in ("proof", "fresh-fit") or not first <= time.time() < end <= 1791273600:
        raise ValueError("prospective separate clock")
    if end - first > (600 if proof else 1800):
        raise ValueError("phase clock cap")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    train = pinned(reg["train"])
    sys.path.insert(0, str(train.parent))
    native = module(reg["native"], "nnue_proof_native")
    contract_path, data = pinned(reg["contract"]), pinned(reg["dataset"])
    contract = json.loads(contract_path.read_bytes())
    if (contract["phase"] != "teacher-bootstrap" or contract["updates"] != 256
            or contract["math"] != native.MATH or contract["dataset_sha256"] != sha(data)
            or contract["original_first_epoch"] != first
            or contract["original_deadline_epoch"] != end):
        raise ValueError("complete production-phase contract")
    out = Path(reg["output"])
    if not out.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM outputs")
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(name, stop, resume=None, audit=False):
        cmd = [sys.executable, str(train), "--contract", str(contract_path),
               "--dataset", str(data), "--output", str(out / name), "--stop", str(stop)]
        if resume:
            cmd += ["--resume", str(resume), "--resume-sha256", sha(resume)]
        if audit:
            cmd += ["--audit-only"]
        remaining = end - time.time()
        if remaining <= 0:
            raise TimeoutError("original phase deadline")
        p = subprocess.run(cmd, capture_output=True, timeout=remaining,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        commands.append(dict(command=cmd, code=p.returncode, stdout=p.stdout.decode(),
                             stderr=p.stderr.decode()))
        if p.returncode:
            raise RuntimeError("child failure preserved")

    try:
        stop = 8 if proof else 256
        run("whole", stop)
        paths = [out / "whole/initial.pt", out / "whole/native.pt"]
        if proof:
            run("pause", 4)
            run("resume", 8, out / "pause/native.pt")
            a = native.load_native(paths[1], contract).native()
            b = native.load_native(out / "resume/native.pt", contract).native()
            if not native.bits_equal(a, b):
                raise ValueError("full model/baseline/Adam/all RNG storage mismatch")
            paths += [out / "pause/initial.pt", out / "pause/native.pt",
                      out / "resume/initial.pt", out / "resume/native.pt"]
        for i, path in enumerate(paths):
            step = native.load_native(path, contract).step
            run(f"audit-{i}", step, path, True)
        if time.time() >= end:
            raise TimeoutError("final deadline")
        result = dict(status="PASS-readonly-loads-and-fixed-phase-not-strength", mode=reg["mode"],
                      full_payload_bits_equal=proof, raw_zip_identity_claimed=False,
                      native_sha256={str(p): sha(p) for p in paths},
                      contract_sha256=sha(contract_path))
    except BaseException as e:
        result = dict(status="FAILED-preserved", error=repr(e))
        raise
    finally:
        result.update(first=first, deadline=end, finished=time.time(), commands=commands)
        (out / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    execute(json.loads(p.parse_args().registration.read_bytes()))
