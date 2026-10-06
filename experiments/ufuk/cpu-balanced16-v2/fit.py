"""Owned fresh48 then two fresh audit-only strict loads within SAME ROOT900."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from support import canonical, publish, registration, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    r, out, guard = registration(a.registration)

    def run(name, resume=None):
        reg = out / (name + "-registration.json")
        publish(reg, dict(r, output=str(out / name)))
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--registration",
            str(reg),
            "--stop",
            "48" if resume is None else str(json.loads(resume.read_bytes())["step"]),
        ]
        if resume is not None:
            cmd += ["--resume", str(resume), "--audit-only"]
        guard()
        with (out / (name + ".log")).open("xb") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            publish(
                out / (name + "-owner.json"),
                dict(
                    pid=proc.pid,
                    command=cmd,
                    startticks=Path(f"/proc/{proc.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19],
                    original_deadline=r["deadline_epoch"],
                ),
            )
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
                raise ValueError("owned child failed " + name)
        guard()
        return out / name

    try:
        fit = run("fresh48")
        payloads = []
        for index, name in enumerate(["initial.json", "native.json"]):
            original = fit / name
            check = run("fresh-load-" + str(index), original) / "native.json"
            if canonical(json.loads(original.read_bytes())) != canonical(
                json.loads(check.read_bytes())
            ):
                raise ValueError("fresh full native storage mismatch")
            payloads.append(
                dict(
                    path=str(original),
                    sha256=sha(original),
                    roundtrip_path=str(check),
                    roundtrip_sha256=sha(check),
                )
            )
        for name in ["initial.json", "native.json", "candidate.json", "result.json"]:
            with (out / name).open("xb") as stream:
                stream.write((fit / name).read_bytes())
        guard()
        publish(
            out / "fresh-loads.json",
            dict(
                schema="balanced-v2-fit-fresh-loads-v1",
                registration_sha256=sha(a.registration),
                payloads=payloads,
                finished=time.time(),
                deadline=r["deadline_epoch"],
                optimizer_updates_in_audit=0,
            ),
        )
    except BaseException as e:
        publish(out / "failure.json", dict(status="INCOMPLETE", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
