"""ROOT-only actual data whole8/pause4/freshresume8 plus six fresh strict loads."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from support import publish, registration, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    r, out, guard = registration(a.registration)
    children = []

    def run(name, stop, resume=None):
        guard()
        child = dict(r, output=str(out / name))
        reg = out / (name + "-registration.json")
        publish(reg, child)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--registration",
            str(reg),
            "--stop",
            str(stop),
        ]
        if resume is not None:
            cmd += ["--resume", str(resume)]
        with (out / (name + ".stdout")).open("xb") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            children.append(proc)
            try:
                while proc.poll() is None:
                    guard()
                    time.sleep(0.05)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
            if proc.returncode:
                raise ValueError("fresh child failed " + name)
        guard()
        return out / name

    try:
        w = run("whole", 8)
        s = run("split", 4)
        v = run("resume", 8, s / "native.json")
        if (w / "native.json").read_bytes() != (v / "native.json").read_bytes():
            raise ValueError("whole8 vs freshresume8 fullnative bytes")
        files = [
            w / "initial.json",
            w / "native.json",
            s / "initial.json",
            s / "native.json",
            v / "initial.json",
            v / "native.json",
        ]
        receipts = []
        for i, f in enumerate(files):
            native = json.loads(f.read_text())
            audit = run("strict-" + str(i), native["step"], f)
            if f.read_bytes() != (audit / "native.json").read_bytes():
                raise ValueError("fresh strict full native roundtrip")
            receipts.append(
                dict(
                    path=str(f),
                    sha256=sha(f),
                    roundtrip_path=str(audit / "native.json"),
                    roundtrip_sha256=sha(audit / "native.json"),
                )
            )
        guard()
        publish(
            out / "qualification.json",
            dict(
                status="PASS-balanced-v2-own-data8-4-fresh8-six-fullnative-loads-not-strength",
                native_payloads=receipts,
                registration_sha256=sha(a.registration),
                finished=time.time(),
                deadline=r["deadline_epoch"],
            ),
        )
    except BaseException as e:
        publish(out / "failure.json", dict(status="INCOMPLETE", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
