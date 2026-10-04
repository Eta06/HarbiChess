"""Owned evaluation utilities; no training controller or MC schemas."""

import hashlib
import json
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

HARD_DEADLINE = 1791170400


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def publish(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def terminate(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def install_interrupt_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def cpu_total_memory():
    path = Path("/sys/fs/cgroup/memory.current")
    if path.exists():
        return (int(path.read_text()), "cgroup.memory.current")
    values = {
        line.split(":")[0]: int(line.split(":")[1].strip().split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    return (values["MemTotal"] - values["MemAvailable"], "system-used-MemTotal-minus-MemAvailable")
