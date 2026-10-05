"""CPU contingency resource support; no strength eligibility."""

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
MEMORY = 15 * 1024**3
DISK = 256 * 1024**2


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


def parse_cgroup_memory(current_text, stat_text, max_text):
    """Estimate nonreclaimable charge; file cache is not private resident memory.

    File-minus-shmem and slab_reclaimable are distinct reclaimable charges.
    This is a cgroup-accounting estimate, not RSS or Linux's working-set metric.
    """
    current = int(current_text.strip())
    stats = {}
    for line in stat_text.splitlines():
        key, value = line.split()
        if key in stats:
            raise ValueError("duplicate cgroup memory.stat key")
        stats[key] = int(value)
    assert current >= 0 and all(value >= 0 for value in stats.values())
    assert {"file", "shmem", "slab_reclaimable", "inactive_file"} <= stats.keys()
    assert stats["shmem"] <= stats["file"]
    maximum = None if max_text.strip() == "max" else int(max_text.strip())
    assert maximum is None or maximum > 0
    reclaimable = stats["file"] - stats["shmem"] + stats["slab_reclaimable"]
    return {
        "metric": "estimated-nonreclaimable-cgroup-charge",
        "current_bytes": current,
        "configured_cgroup_max_bytes": maximum,
        "file_bytes": stats["file"],
        "shmem_bytes": stats["shmem"],
        "slab_reclaimable_bytes": stats["slab_reclaimable"],
        "inactive_file_bytes": stats["inactive_file"],
        "working_set_current_minus_inactive_file_bytes": max(0, current - stats["inactive_file"]),
        "estimated_nonreclaimable_bytes": max(0, current - reclaimable),
        "nonreclaimable_ceiling_bytes": MEMORY,
        "total_charge_ceiling_bytes": 16 * 1024**3,
    }


def memory_snapshot():
    root = Path("/sys/fs/cgroup")
    if (root / "memory.current").exists():
        return parse_cgroup_memory(
            (root / "memory.current").read_text(),
            (root / "memory.stat").read_text(),
            (root / "memory.max").read_text(),
        )
    fields = {
        line.split(":")[0]: int(line.split(":")[1].strip().split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    return {
        "metric": "system-MemTotal-minus-MemAvailable-no-cgroup",
        "current_bytes": fields["MemTotal"] - fields["MemAvailable"],
        "configured_cgroup_max_bytes": None,
        "estimated_nonreclaimable_bytes": fields["MemTotal"] - fields["MemAvailable"],
        "nonreclaimable_ceiling_bytes": MEMORY,
        "total_charge_ceiling_bytes": 16 * 1024**3,
    }


def memory_used():
    return memory_snapshot()["estimated_nonreclaimable_bytes"]


def guard(deadline, output):
    snapshot = memory_snapshot()
    snapshot.update(
        observed_epoch=time.time(),
        deadline_epoch=deadline,
        disk_free_bytes=shutil.disk_usage(output).free,
        disk_min_bytes=DISK,
    )
    violations = []
    if snapshot["observed_epoch"] >= deadline:
        violations.append("original-wholedeadline-exhausted")
    if snapshot["estimated_nonreclaimable_bytes"] > MEMORY:
        violations.append("nonreclaimable-charge-above15GiB")
    if snapshot["current_bytes"] > 16 * 1024**3:
        violations.append("total-cgroup-charge-above16GiB")
    if snapshot["disk_free_bytes"] < DISK:
        violations.append("free-disk-below256MiB")
    if violations:
        snapshot["violations"] = violations
        raise RuntimeError(json.dumps(snapshot, sort_keys=True))


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
