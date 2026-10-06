"""Prospectively pinned ROOT-only execution/resource contract."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

END = 1791273600


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def publish(p, x):
    with Path(p).open("xb") as f:
        f.write(canonical(x) + b"\n")


def load(p, s, name):
    if sha(p) != s:
        raise ValueError("helper SHA")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def registration(path):
    r = json.loads(Path(path).read_text())
    if r["schema"] != "own-selective-q-ROOT-registration-v1" or r["status"] != "registered":
        raise ValueError("ROOT registered only")
    if r["first_epoch"] is None or not r["first_epoch"] <= time.time() < r["deadline_epoch"] <= min(
        END, r["first_epoch"] + 900
    ):
        raise ValueError("original900 no reset")
    if r["seed"] not in [20262905, 20262906]:
        raise ValueError("two fixed data seeds")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=r["source_repo"], text=True
        ).strip()
        != r["source_commit"]
    ):
        raise ValueError("source pin")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=r["source_repo"], text=True
    ).strip():
        raise ValueError("clean source")
    for helper in Path(__file__).parent.glob("*.py"):
        if helper.name.startswith("test_"):
            continue
        if r["helper_sha256"].get(str(helper.resolve())) != sha(helper):
            raise ValueError("full local helper closure")
    for p, s in r["helper_sha256"].items():
        if sha(p) != s:
            raise ValueError("exact closure")
    for x in r["inputs"].values():
        if sha(x["path"]) != x["sha256"]:
            raise ValueError("input SHA")
    output = Path(r["output"])
    if not output.resolve().is_relative_to(Path("/dev/shm")):
        raise ValueError("RAM artifact scope")
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {r["cpu_core"]})
    sys.path.insert(0, str(Path(r["source_repo"]) / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        if time.time() >= r["deadline_epoch"]:
            raise TimeoutError("original900")
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256MiB")
        if sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) > 16 * 2**20:
            raise RuntimeError("ownRAM16MiB")

    guard()
    return r, output, guard
