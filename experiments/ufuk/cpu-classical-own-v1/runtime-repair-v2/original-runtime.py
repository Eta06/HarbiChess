"""Explicit one-CPU prospective resource checks; external owner remains required."""

import os
import shutil
import time
from pathlib import Path

HARD_END = 1791273600.0


def guard(deadline, workspace, artifacts):
    if not time.time() < deadline <= HARD_END:
        raise TimeoutError("original deadline or hard08 exhausted")
    if shutil.disk_usage(workspace).free < 256 * 1024**2:
        raise RuntimeError("workspace256MiB floor")
    if sum(p.stat().st_size for p in Path(artifacts).iterdir() if p.is_file()) >= 16 * 1024**2:
        raise RuntimeError("artifact16MiB ceiling")
    root = Path("/sys/fs/cgroup")
    if (root / "memory.current").is_file():
        current = int((root / "memory.current").read_text())
        stat = {
            k: int(v)
            for k, v in (r.split() for r in (root / "memory.stat").read_text().splitlines())
        }
        if max(0, current - stat.get("inactive_file", 0)) > 15 * 1024**3:
            raise RuntimeError("inactive-file-v1 working-set15GiB ceiling")


def affinity(core):
    if core not in os.sched_getaffinity(0):
        raise ValueError("CPU outside allowed affinity")
    os.sched_setaffinity(0, {core})
