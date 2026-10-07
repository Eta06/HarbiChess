"""SHA-bound source namespaces, ROOT clocks and existing CPU resource semantics."""

import contextlib
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

END = 1791448916.685839
CORE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def pinned(value):
    p = Path(value["path"])
    if not p.is_file() or p.is_symlink() or sha(p) != value["sha256"]:
        raise ValueError("immutable exact file SHA")
    return p


def read(value):
    return json.loads(pinned(value).read_bytes())


def pins_tree(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            pinned(value)
        for x in value.values():
            pins_tree(x)
    elif isinstance(value, list):
        for x in value:
            pins_tree(x)


def module(value, name):
    spec = importlib.util.spec_from_file_location(name, pinned(value))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def namespace(directory, inventories):
    directory = Path(directory).resolve()
    files = {}
    for value in inventories:
        inventory = read(value)
        if "current_source_files" in inventory:
            for path, digest in inventory["current_source_files"].items():
                pinned(dict(path=path, sha256=digest))
                p = Path(path).resolve()
                if p.parent == directory:
                    files[p.name] = digest
        else:
            for name, record in inventory["files"].items():
                digest = record if isinstance(record, str) else record["sha256"]
                if Path(name).name != name:
                    raise ValueError("source basename")
                pinned(dict(path=str(directory / name), sha256=digest))
                files[name] = digest
    names = {p.stem for p in directory.glob("*.py")}
    old = {name: sys.modules.pop(name, None) for name in names}
    before = list(sys.path)
    sys.path.insert(0, str(directory))
    try:
        yield files
    finally:
        sys.path[:] = before
        for name in names:
            sys.modules.pop(name, None)
            if old[name] is not None:
                sys.modules[name] = old[name]


def clock(spec, cap):
    a, z, o = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) for v in (a, z, o)
    ) or not a < z <= min(a + cap, o, END):
        raise ValueError("ROOT separate prospective clock")


def guard(spec, cap):
    clock(spec, cap)
    core = Path(spec["core_repo"])
    if (
        spec["core_commit"] != CORE
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
        != CORE
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True)
    ):
        raise ValueError("clean original6fcc core")
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)
    monotonic = time.monotonic() + spec["deadline"] - time.time()

    def check():
        memory.check()
        if not spec["first"] <= time.time() < spec["deadline"] or time.monotonic() >= monotonic:
            raise TimeoutError("original phase clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256MiB")

    check()
    return check


def publish(path, value):
    with Path(path).open("xb") as f:
        f.write(canonical(value) + b"\n")
