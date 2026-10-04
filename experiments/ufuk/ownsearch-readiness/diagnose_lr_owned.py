"""Own the single duplicate-only LR diagnosis and its fixed shared 300s budget."""

import argparse
import json
import os
import time
from pathlib import Path

from qualify_cuda_owned import check_source, publish, run_owned, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "profile", "output", "config", "registration", "weights", "book"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    if not __debug__ or not 0 < a.deadline_epoch - started <= 300:
        raise ValueError("Requires one original shared absolute 300s ceiling")
    check_source(a.checkout, a.source_commit)
    reg = json.loads(a.registration.read_text())
    if sha(Path(__file__)) != reg["controller_sha256"]:
        raise ValueError("Prospective controller pin differs")
    a.output.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(a.checkout / "src"), PYTHONOPTIMIZE="0",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    helper = Path(__file__).with_name("diagnose_lr.py")
    command = ["/usr/bin/python3", str(helper)]
    for name in ("checkout", "profile", "config", "registration", "weights", "book"):
        command += ["--" + name, str(getattr(a, name))]
    command += ["--output", str(a.output / "diagnosis"), "--source-commit",
                a.source_commit, "--deadline-epoch", str(a.deadline_epoch)]
    result = dict(schema="owned300-duplicate-lr-diagnosis-v1", status="failed-preserved",
                  source_commit=a.source_commit, registration_sha256=sha(a.registration),
                  started_epoch=started, absolute_deadline_epoch=a.deadline_epoch)
    try:
        result["phase"] = run_owned("duplicate-only-fitting", command, cwd=a.checkout,
                                    env=env, output=a.output, deadline=a.deadline_epoch)
        receipt = a.output / "diagnosis/result.json"
        result["diagnosis"] = json.loads(receipt.read_text())
        result["diagnosis_sha256"] = sha(receipt)
        if result["diagnosis"]["status"] != "completed-duplicate-only-no-strength-claim":
            raise ValueError("Diagnosis did not finish within original ceiling")
        result["status"] = "completed-duplicate-only-no-strength-claim"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - started
        publish(a.output / "result.json", result)


if __name__ == "__main__":
    main()
