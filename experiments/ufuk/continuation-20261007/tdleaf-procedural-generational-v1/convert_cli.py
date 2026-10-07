"""ROOT-only bounded full-ledger conversion; no search, inference or SGD."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import convert
from parent_bridge import canonical, read, sha

END = 1791448916.685839


def execute(spec_path, clock_path, output):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_bytes())
    clock = json.loads(Path(clock_path).read_bytes())
    if (
        clock["schema"] != "procedural-generational-tdleaf-conversion-clock-v1"
        or clock["helper_sha256"] != sha(__file__)
        or clock["conversion_seal_sha256"] != sha(spec_path)
        or not clock["first"]
        <= time.time()
        < clock["deadline"]
        <= min(clock["first"] + 600, clock["operator_end_epoch"], END)
    ):
        raise ValueError("original ROOT conversion clock/source/seal")
    reg = read(spec["registration"])
    if reg["operator_end_epoch"] != clock["operator_end_epoch"]:
        raise ValueError("same original operator scope")
    sys.path.insert(0, reg["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    os.sched_setaffinity(0, {clock["cpu_core"]})

    def guard():
        budget.check()
        if time.time() >= clock["deadline"] or shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("original conversion deadline/disk floor")

    guard()
    data, provenance = convert.convert_sealed(spec)
    guard()
    out = Path(output).resolve()
    if not out.is_relative_to("/dev/shm") or out.exists():
        raise ValueError("new RAM output only")
    payloads = {"dataset.json": canonical(data), "provenance.json": canonical(provenance)}
    if any(len(v) > 8 * 2**20 for v in payloads.values()):
        raise ValueError("each converted artifact8MiB")
    out.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        with (out / name).open("xb") as f:
            f.write(value)
    guard()
    result = dict(
        status="PASS-generational-TDLeaf-full-ledger-conversion-not-strength",
        first=clock["first"],
        deadline=clock["deadline"],
        finished=time.time(),
        clock_sha256=sha(clock_path),
        helper_sha256=sha(__file__),
        conversion_seal_sha256=sha(spec_path),
        dataset_sha256=sha(out / "dataset.json"),
        provenance_sha256=sha(out / "provenance.json"),
    )
    with (out / "result.json").open("xb") as f:
        f.write(canonical(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--clock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    execute(a.seal, a.clock, a.output)
