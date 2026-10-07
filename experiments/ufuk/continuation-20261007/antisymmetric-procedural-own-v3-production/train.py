"""ROOT actual paired own64 full resume/audit; no old NNUE fullresume or teacher."""

import argparse
import json
from pathlib import Path

import torch
from native import Learner, load_native
from support import guard, pinned, pins_tree, read, sha

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["contract", "dataset", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    c = json.loads(a.contract.read_bytes())
    if c.get("synthetic_only"):
        raise ValueError("synthetic production-math fixture is not an actual phase")
    check = guard(c, 600 if c["mode"] == "proof" else 1800)
    if c["phase"] != "antisymmetric-procedural-own-learning-v3" or c["dataset_sha256"] != sha(
        a.dataset
    ):
        raise ValueError("exact new paired phase/data")
    pins_tree(c)
    required = {
        "model.py",
        "native.py",
        "train.py",
        "support.py",
        "zero_parent.py",
        "initialize.py",
        "convert.py",
        "adapter.py",
        "contracts.py",
        "prove.py",
        "admission.py",
        "compiled_evaluator.py",
    }
    if set(c["source_refs"]) != required:
        raise ValueError("exact complete runtime inventory")
    for n, v in c["source_refs"].items():
        if v["path"] != str(HERE / n) or sha(HERE / n) != v["sha256"]:
            raise ValueError("complete actual new runtime source closure")
    data = read(c["dataset"])
    if (
        data["schema"] != "own-same-board-antisymmetric-forensic-data-v3"
        or data["phase"] != c["phase"]
        or len(data["rows"]) != 1024
    ):
        raise ValueError("paired common own1024 data")
    learner = (
        load_native(pinned(dict(path=str(a.resume), sha256=a.resume_sha256)), c)
        if a.resume
        else Learner(c)
    )
    if a.audit_only:
        check()
        print(
            json.dumps(dict(status="PASS-strict-antisymmetric-native-readonly", step=learner.step))
        )
        return
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM native outputs")
    a.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (a.output / name).open("xb") as f:
            torch.save(value, f)
        if (a.output / name).stat().st_size > 8 * 2**20:
            raise ValueError("native8MiB")

    save("initial.pt", learner.native())
    learner.advance(data["rows"], a.stop, check)
    check()
    save("native.pt", learner.native())
    if learner.step == 64:
        save("candidate.pt", learner.candidate())


if __name__ == "__main__":
    main()
