"""Real final16384 data whole8/pause4/freshresume8, six strict fresh native loads."""

import argparse
import json
import sys
import time
from pathlib import Path

from train import sha

# ROOT binds the existing ownership/publish helper closure in the protocol.
# It is imported only after immutable dependency validation.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("protocol", "config", "journal", "source-repo", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args()
    if (
        not a.first <= time.time() < a.deadline
        or a.deadline != a.first + 600
        or a.deadline > 1791273600.0
    ):
        raise ValueError("observed original600/hard08clock required")
    if sha(a.protocol) != a.protocol_sha256:
        raise ValueError("qualification protocol SHA differs")
    protocol = json.loads(a.protocol.read_text())
    if protocol["schema"] != "own-quiet-ordering-training-protocol-v1":
        raise ValueError("qualification schema differs")
    for name, expected in protocol["ordering_helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("actual ordering proof helper differs")
    old = Path(protocol["classical_helper_directory"])
    for name, expected in protocol["classical_helper_sha256"].items():
        if sha(old / name) != expected:
            raise ValueError("immutable qualifier dependencies differ")
    sys.path.append(str(old))
    from qualify_actor_v3 import run_owned
    from train_v3 import publish

    a.output.mkdir(exist_ok=False)
    base = [sys.executable, str(Path(__file__).with_name("train.py"))]
    for k in ("protocol", "config", "journal", "source-repo"):
        base += ["--" + k, str(getattr(a, k.replace("-", "_")))]
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
    from runtime import guard

    def parent_guard():
        return guard(a.deadline, a.source_repo, a.output)

    whole = a.output / "whole"
    split = a.output / "split"
    run_owned(
        [*base, "--output", str(whole), "--stop-at", "8"],
        a.deadline,
        a.output,
        "whole8",
        parent_guard,
    )
    run_owned(
        [*base, "--output", str(split), "--stop-at", "4"],
        a.deadline,
        a.output,
        "pause4",
        parent_guard,
    )
    pause = split / "step-00000004.json"
    run_owned(
        [
            *base,
            "--output",
            str(split),
            "--stop-at",
            "8",
            "--resume",
            str(pause),
            "--resume-sha256",
            sha(pause),
        ],
        a.deadline,
        a.output,
        "resume8",
        parent_guard,
    )
    manifests = []
    for owner, step in [(whole, 0), (whole, 4), (whole, 8), (split, 0), (split, 4), (split, 8)]:
        native = owner / f"step-{step:08d}.json"
        run_owned(
            [
                *base,
                "--output",
                str(owner),
                "--resume",
                str(native),
                "--resume-sha256",
                sha(native),
                "--audit-only",
            ],
            a.deadline,
            a.output,
            f"load-{owner.name}-{step}",
            parent_guard,
        )
        manifests.append(dict(path=str(native), sha256=sha(native), step=step))
    if (whole / "step-00000008.json").read_bytes() != (split / "step-00000008.json").read_bytes():
        raise ValueError("full Adam/theta/RNG/contract restart differs")
    parent_guard()
    if time.time() >= a.deadline:
        raise TimeoutError("original qualification600 includes strict-load audit")
    receipt = dict(
        finished_epoch=time.time(),
        schema="own-quiet-ordering-native-qualification-v1",
        status="PASS-quiet-order-whole8-pause4-freshresume8-not-strength",
        original_first=a.first,
        original_deadline=a.deadline,
        full_native_bytes_equal=True,
        strict_fresh_native_loads=manifests,
        protocol_sha256=a.protocol_sha256,
    )
    publish(a.output / "result.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
