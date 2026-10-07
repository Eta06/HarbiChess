"""ROOT actual immutable900 allhistory audit then six actual search replays."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/workspace/work/harbichess/continuation-20261007")
STUDY = BASE / "ranking-known160-runtime"
DATA = Path("/dev/shm/harbichess-continuation-20261007/ranking-known160-data")
OUT = BASE / "ranking-root-independent-audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    q = json.loads((STUDY / "protocol.json").read_bytes())
    first = time.time()
    deadline = min(first + 900, q["audit_deadline_epoch"], q["ROOToperator_end_epoch"])
    clock = dict(
        schema="NNUE-strength-root-independent-audit-clock-v2",
        first=first,
        deadline=deadline,
        cpu_core=0,
        protocol_sha256=sha(STUDY / "protocol.json"),
    )
    (OUT / "clock.json").write_text(json.dumps(clock, sort_keys=True, indent=2) + "\n")
    os.sched_setaffinity(0, {0})
    commands = [
        [
            sys.executable,
            str(STUDY / "audit_known160_v2.py"),
            "--protocol",
            str(STUDY / "protocol.json"),
            "--arena",
            str(DATA / "arena"),
            "--output",
            str(OUT / "fullhistory.json"),
        ],
        [
            sys.executable,
            str(STUDY / "replay_six.py"),
            "--protocol",
            str(STUDY / "protocol.json"),
            "--profile",
            str(DATA / "profile.json"),
            "--arena",
            str(DATA / "arena"),
            "--clock",
            str(OUT / "clock.json"),
            "--output",
            str(OUT / "six-actual-replays.json"),
        ],
    ]
    receipt = dict(
        status="running",
        first=first,
        deadline=deadline,
        helper_sha256=sha(Path(__file__)),
        pid=os.getpid(),
        commands=commands,
        completed=[],
    )
    (OUT / "registration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for index, command in enumerate(commands):
            with (
                (OUT / f"{index}.stdout.log").open("xb") as out,
                (OUT / f"{index}.stderr.log").open("xb") as err,
            ):
                subprocess.run(
                    command,
                    check=True,
                    timeout=max(0.001, deadline - time.time()),
                    stdout=out,
                    stderr=err,
                )
            receipt["completed"].append(index)
        if time.time() >= deadline:
            raise TimeoutError("same original900 exceeded")
        receipt.update(
            status="PASS-integrity-and-six-actual-not-strength",
            outputs={
                name: sha(OUT / name)
                for name in ("fullhistory.json", "six-actual-replays.json")
            },
        )
    except Exception as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished"] = time.time()
        (OUT / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
