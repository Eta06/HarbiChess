"""ROOT read-only admission in an isolated interpreter; no forward/search/SGD."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import parent_bridge as bridge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clock", type=Path, required=True)
    a = p.parse_args()
    seal = {"path": str(a.seal.resolve()), "sha256": bridge.sha(a.seal)}
    spec = bridge.read(seal)
    clock = json.loads(a.clock.read_bytes())
    if (
        clock["schema"] != "human-prior-own-parent-readonly-admission-clock-v1"
        or clock["helper_sha256"] != bridge.sha(__file__)
        or clock["seal_sha256"] != seal["sha256"]
        or not clock["first"]
        <= time.time()
        < clock["deadline"]
        <= min(clock["first"] + 600, clock["operator_end_epoch"], bridge.END)
    ):
        raise ValueError("ROOT separate600 read-only parent-admission clock")
    contract = bridge.validate_metadata(spec)
    sys.path.insert(0, contract["core_source_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)
    memory.check()
    model = bridge.pinned(spec["parent_model_helper"])
    source = bridge.pinned(spec["parent_native_helper"])
    sys.path.insert(0, str(model.parent))
    sys.modules.pop("model", None)

    # Isolated CLI, so no current-child native is imported or monkeypatched.
    def load(path, name):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m

    load(model, "model")
    original = load(source, "original_parent_native")
    import torch

    torch.set_num_threads(1)
    _, result = bridge.admit(
        spec,
        original,
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
    )
    memory.check()
    if not time.time() < clock["deadline"]:
        raise TimeoutError("same original read-only admission clock")
    result.update(
        seal_sha256=seal["sha256"],
        bridge_sha256=bridge.sha(bridge.__file__),
        clock_sha256=bridge.sha(a.clock),
        clock={"path": str(a.clock.resolve()), "sha256": bridge.sha(a.clock)},
        first=clock["first"],
        deadline=clock["deadline"],
        finished=time.time(),
    )
    with a.output.open("xb") as f:
        f.write(bridge.canonical(result) + b"\n")


if __name__ == "__main__":
    main()
