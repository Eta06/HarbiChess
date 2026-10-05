"""Owned CUDA infrastructure qualification: units180s, CLI180s; no strength eligibility."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

E8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
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


def kill_group(child):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def run_owned(name, command, *, cwd, env, output, deadline):
    guard(deadline, output)
    record = dict(
        phase=name,
        command=command,
        started_epoch=time.time(),
        absolute_deadline_epoch=deadline,
        status="failed",
    )
    child = None
    try:
        with (
            (output / (name + ".stdout.log")).open("x") as out,
            (output / (name + ".stderr.log")).open("x") as err,
        ):
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            record["owned_pid"] = record["owned_process_group"] = child.pid
            publish(
                output / (name + ".owner.json"),
                dict(
                    pid=child.pid,
                    owned_process_group=child.pid,
                    deadline_epoch=deadline,
                ),
            )
            while child.poll() is None:
                guard(deadline, output)
                time.sleep(min(0.1, max(0.001, deadline - time.time())))
        record["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"{name} returned {child.returncode}")
        guard(deadline, output)
        record["status"] = "pass"
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        if child is not None:
            # Clean descendants even if their direct parent exited or failed.
            kill_group(child)
            record["returncode"] = child.returncode
        record["finished_epoch"] = time.time()
        record["whole_seconds"] = record["finished_epoch"] - record["started_epoch"]
        publish(output / (name + ".invocation.json"), record)
    return record


def check_source(checkout, source):
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        != source
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        ).strip()
    ):
        raise ValueError("requires exact CLEANnew ownsearch producer")
