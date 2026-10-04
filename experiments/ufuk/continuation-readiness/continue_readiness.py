"""Wait for original MC checks, then start two prospectively bounded readiness jobs."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

END = 1791170400
WAIT_END = 1791155520
MC_SHA = "686265b09284d1e8c879406510910991d2d8f1a71bf60b610372eac00e76e100"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    spec_path = args.inputs / "spec.json"
    assert sha(spec_path) == args.spec_sha256
    spec = json.loads(spec_path.read_text())
    assert spec["supervisor_sha256"] == sha(__file__)
    args.root.mkdir(exist_ok=False)
    mc = Path("/content/harbichess-runs/fullgame-confirmation3-posttraining-recovery-v2")
    assert sha(
        "/content/harbichess-fullgame-method2-inputs/confirmation3/"
        "posttraining-recovery-v2/orchestrate.py"
    ) == MC_SHA
    names = ["replay-20261205", "replay-20261206", "eligibility", "cuda-parity", "latency"]
    processes = {}
    try:
        while True:
            assert time.time() < WAIT_END, "Original MC checks missed readiness wait ceiling"
            try:
                receipts = []
                for name in names:
                    path = mc / (name + "-process-result.json")
                    result = json.loads(path.read_text())
                    assert result["returncode"] == 0
                    assert result["finished_epoch"] <= result["deadline_epoch"]
                    receipts.append(dict(path=str(path), sha256=sha(path)))
                assert (mc / "latency.json").is_file()
                break
            except (FileNotFoundError, json.JSONDecodeError):
                failure = mc / "orchestration-failure.json"
                assert not failure.exists(), "Original MC readiness failed; preserve and diagnose"
                time.sleep(5)
        barrier = dict(
            coordinator_sha256=MC_SHA,
            process_receipts=receipts,
            latency_receipt=dict(path=str(mc / "latency.json"), sha256=sha(mc / "latency.json")),
        )
        publish(args.root / "mc-completion-barrier.json", barrier)
        for name in ("own4-E1-readonly-requalification", "search-acting-v2-CUDA"):
            job = spec["jobs"][name]
            for path, digest in job["frozen_sha256"].items():
                assert sha(path) == digest
            started = time.time()
            deadline = started + job["whole_seconds"]
            assert deadline < END
            command = list(job["command"])
            if name == "own4-E1-readonly-requalification":
                protocol = Path(job["protocol"])
                data = json.loads(protocol.read_text())
                assert data["started_epoch"] == "ACTUAL_FIRST_CLOCK_AT_LAUNCH"
                data["started_epoch"] = started
                data["prospective_protocol_sha256"] = sha(protocol)
                manifest = protocol.parent / "actual-launch-manifest.json"
                publish(manifest, data)
                command += ["--manifest", str(manifest), "--manifest-sha256", sha(manifest)]
            else:
                command += ["--deadline-epoch", str(deadline)]
            env = dict(
                os.environ, PYTHONPATH=job["checkout"] + "/src", PYTHONOPTIMIZE="0",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            with (
                (args.root / (name + ".stdout.log")).open("x") as out,
                (args.root / (name + ".stderr.log")).open("x") as err,
            ):
                child = subprocess.Popen(
                    command, cwd=job["checkout"], env=env, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err, start_new_session=True,
                )
            processes[name] = child
            publish(args.root / (name + "-owner.json"), dict(
                pid=child.pid, owned_process_group=child.pid, started_epoch=started,
                absolute_deadline_epoch=deadline, command=command,
                MC_readiness_barrier_sha256=sha(args.root / "mc-completion-barrier.json"),
            ))
        # Child controllers independently own and enforce their original deadlines.
        # This supervisor never retries or modifies their artifacts or clocks.
        for name, child in processes.items():
            returncode = child.wait()
            publish(args.root / (name + "-process-result.json"), dict(
                returncode=returncode, finished_epoch=time.time(),
            ))
        publish(args.root / "result.json", dict(
            status="readiness-children-finished-not-strength", finished_epoch=time.time(),
            children_returncode={name: p.returncode for name, p in processes.items()},
        ))
    except BaseException as error:
        publish(args.root / "failure.json", dict(
            status="failed-preserved-no-retry", error_type=type(error).__name__,
            finished_epoch=time.time(),
        ))
        raise


if __name__ == "__main__":
    main()
