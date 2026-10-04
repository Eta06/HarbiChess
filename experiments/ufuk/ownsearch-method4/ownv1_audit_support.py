"""Exact qualified readiness resource/source/atomic publication primitives."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

MEMORY = 64 * 1024**3
DISK = 8 * 1024**3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, data):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def memory_used():
    path = Path("/sys/fs/cgroup/memory.current")
    if path.exists():
        return int(path.read_text())
    fields = {
        line.split(":")[0]: int(line.split(":")[1].strip().split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    return fields["MemTotal"] - fields["MemAvailable"]


def guard(deadline, output):
    if time.time() >= deadline:
        raise TimeoutError("original wholedeadline exhausted")
    if memory_used() > MEMORY or shutil.disk_usage(output).free < DISK:
        raise RuntimeError("registered64GiB/8GiB resource ceiling")


def check_source(checkout, source):
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        != source
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        ).strip()
    ):
        raise ValueError("requires exact CLEANnew ownsearch producer")
