"""ROOT future native qualification; no process launched on import."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from train_actions import sha


def main():
    p = argparse.ArgumentParser()
    for name in ["protocol", "labels", "output", "source-repo"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args()
    if not a.first <= time.time() < a.deadline <= min(a.first + 600, 1791273600):
        raise ValueError("new qualification600 clock")
    a.output.mkdir(exist_ok=False)
    base = [sys.executable, str(Path(__file__).with_name("train_actions.py"))]
    for name in ["protocol", "labels", "source-repo"]:
        base += ["--" + name, str(getattr(a, name.replace("-", "_")))]
    base += [
        "--protocol-sha256",
        a.protocol_sha256,
        "--seed",
        str(a.seed),
        "--cpu-core",
        str(a.cpu_core),
        "--deadline",
        str(a.deadline),
    ]
    commands = []

    def run(extra, tag):
        command = base + extra
        commands.append(command)
        with (
            (a.output / (tag + ".stdout")).open("xb") as out,
            (a.output / (tag + ".stderr")).open("xb") as err,
        ):
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            stat = Path(f"/proc/{process.pid}/stat").read_text()
            witness = stat[stat.rfind(")") + 2 :].split()[19]
            try:
                process.wait(timeout=max(0.001, a.deadline - time.time()))
                if process.returncode != 0:
                    raise RuntimeError("owned qualification child failed; logs preserved")
            finally:
                if process.poll() is None:
                    current = Path(f"/proc/{process.pid}/stat").read_text()
                    if current[current.rfind(")") + 2 :].split()[19] == witness:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=0.5)

    whole, split = a.output / "whole", a.output / "split"
    run(["--output", str(whole), "--stop-at", "8"], "whole8")
    run(["--output", str(split), "--stop-at", "4"], "pause4")
    pause = split / "step-00000004.json"
    run(
        [
            "--output",
            str(split),
            "--stop-at",
            "8",
            "--resume",
            str(pause),
            "--resume-sha256",
            sha(pause),
        ],
        "freshresume8",
    )
    for step in [0, 4, 8]:
        name = f"step-{step:08d}.json"
        if (whole / name).read_bytes() != (split / name).read_bytes():
            raise ValueError(
                "all params/Adam/global-sampler RNG/counter/contract native bytes differ"
            )
        for label, directory in [("whole", whole), ("split", split)]:
            native = directory / name
            run(
                [
                    "--output",
                    str(directory),
                    "--audit-only",
                    "--resume",
                    str(native),
                    "--resume-sha256",
                    sha(native),
                ],
                f"freshload-{label}-{step}",
            )
    if time.time() >= a.deadline:
        raise TimeoutError("original qualification600 expired")
    (a.output / "result.json").write_text(
        json.dumps(
            dict(
                schema="own-action-native-qualification-v1",
                status="PASS-six-native-freshloads-and-8-4-fresh8-not-strength",
                first=a.first,
                deadline=a.deadline,
                finished=time.time(),
                commands=commands,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
