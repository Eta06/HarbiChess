"""Operator allocation-window/disk guard: exact registered PID identities only."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from contextlib import suppress
from pathlib import Path


def proc(pid):
    try:
        words = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return {"pid": pid, "state": words[0], "parent": int(words[1]), "ticks": words[19]}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def matches(row):
    p = proc(row["pid"])
    return p is not None and p["ticks"] == str(row["ticks"]) and p["state"] != "Z"


def stop_owned(owner):
    root = {"pid": int(owner["pid"]), "ticks": str(owner["start_ticks"])}
    if not matches(root):
        return []
    os.kill(root["pid"], signal.SIGSTOP)
    allproc = {
        int(p.name): proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdecimal()
    }
    known = {root["pid"]}
    descendants = []
    for _ in range(32):
        new = [p for p in allproc.values() if p and p["parent"] in known and p["pid"] not in known]
        if not new:
            break
        descendants.extend(new)
        known.update(p["pid"] for p in new)
    stopped = []
    for row in reversed(descendants):
        if matches(row):
            try:
                os.kill(row["pid"], signal.SIGTERM)
                stopped.append(row)
            except ProcessLookupError:
                pass
    if matches(root):
        os.kill(root["pid"], signal.SIGTERM)
        os.kill(root["pid"], signal.SIGCONT)
        stopped.append(root)
    end = time.monotonic() + 5
    while time.monotonic() < end and any(matches(p) for p in stopped):
        time.sleep(0.1)
    for row in stopped:
        if matches(row):
            with suppress(ProcessLookupError):
                os.kill(row["pid"], signal.SIGKILL)
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--disk-floor-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        now = time.time()
        free = shutil.disk_usage(args.catalog.parent).free
        reason = (
            "technical-operator-window-expired"
            if now >= args.deadline_epoch
            else "disk-floor"
            if free < args.disk_floor_bytes
            else None
        )
        if reason:
            records = []
            for name in json.loads(args.catalog.read_text())["owner_receipts"]:
                path = Path(name)
                if not path.is_file():
                    continue
                owner = json.loads(path.read_text())
                stopped = stop_owned(owner)
                records.append(
                    {
                        "owner_receipt": str(path),
                        "owner_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "stopped": stopped,
                    }
                )
            receipt = {
                "reason": reason,
                "observed_epoch": now,
                "deadline_epoch": args.deadline_epoch,
                "free_bytes": free,
                "records": records,
                "GPU_used": False,
                "scope": (
                    "only exact registered PID/startticks descendants; "
                    "no checkpoint rewrites or original timer resets"
                ),
            }
            with args.output.open("x") as out:
                json.dump(receipt, out, indent=2)
                out.write("\n")
            return
        if args.once:
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
