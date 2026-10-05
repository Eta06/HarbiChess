"""Root frozen CPU fresh-fit controller, fixed endpoints and strict native audits."""

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

REGPATH = Path(sys.argv[1])
REG = json.loads(REGPATH.read_text())
BASE = REGPATH.parent
END = REG["deadline_epoch"]
FIRST = REG["first_epoch"]
RAM = Path(REG["ram_stage"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert FIRST <= time.time() < END <= 1791273600
assert END - FIRST == 1800
assert sha(__file__) == REG["controller_sha256"]
os.sched_setaffinity(0, {REG["cpu_core"]})
RAM.mkdir(exist_ok=False)
for path, expected in REG["input_pins"].items():
    assert sha(path) == expected, path
ENV = {
    **os.environ,
    "PYTHONPATH": REG["checkout"] + "/src",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
rows = []
child = None


def guard():
    if time.time() >= END:
        raise TimeoutError("original shared1800 fresh-fit clock exhausted")
    if shutil.disk_usage("/workspace").free < 268435456:
        raise RuntimeError("workspace floor breached")
    if (
        sum(p.stat().st_size for p in RAM.rglob("*") if p.is_file())
        > REG["RAMstage_limit_bytes"]
    ):
        raise RuntimeError("sealed RAM stage cap breached")


def invoke(command, label):
    global child
    guard()
    with (
        (BASE / (label + ".stdout.log")).open("x") as out,
        (BASE / (label + ".stderr.log")).open("x") as err,
    ):
        child = subprocess.Popen(
            command,
            cwd=REG["checkout"],
            env=ENV,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        while child.poll() is None:
            guard()
            time.sleep(0.5)
        code = child.returncode
        child = None
    guard()
    if code:
        raise RuntimeError(f"{label} failed with code {code}")


try:
    for job in REG["jobs"]:
        guard()
        started = time.time()
        tag = job["tag"]
        invoke(job["command"], tag + "-fit")
        output = Path(job["output"])
        results = sorted(output.glob("result-step-*-invocation-*.json"))
        assert len(results) == 1
        result = json.loads(results[0].read_text())
        assert result["status"] == "completed-fit-not-strength"
        accepted = result["accepted_updates"]
        derived = result.get("derived_update_budget", result.get("derived_updates"))
        assert accepted == derived == job["expected_updates"]
        contract = result["contract"]
        assert contract["journal_sha256"] == job["journal_sha256"]
        actual_data = contract.get(
            "common_mc_dataset_sha256",
            contract.get("common_dataset_sha256", result.get("dataset_sha256")),
        )
        assert actual_data == job["expected_common_dataset_sha256"]
        for step in (0, accepted):
            command = [
                *job["command"],
                "--resume",
                str(output / "checkpoints" / f"step-{step:08d}"),
                "--audit-only",
            ]
            invoke(command, tag + f"-strict-native-{step}")
        candidate = output / "candidate.safetensors"
        assert (
            candidate.is_file()
            and candidate.stat().st_size <= job["max_each_artifact_bytes"]
        )
        guard()
        record = {
            "tag": tag,
            "status": "PASS-final-fit-and-two-strict-native-loads-not-strength",
            "started_epoch": started,
            "finished_epoch": time.time(),
            "candidate_sha256": sha(candidate),
            "candidate_bytes": candidate.stat().st_size,
            "result_path": str(results[0]),
            "result_sha256": sha(results[0]),
            "result": result,
        }
        rows.append(record)
        print(tag, accepted, record["candidate_sha256"], flush=True)
    guard()
    status = "PASS-all-registered-fits-and-native-audits-not-strength"
except BaseException as exc:
    status = "failed-preserved"
    error = repr(exc)
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
result = {
    "status": status,
    "rows": rows,
    "original_first_epoch": FIRST,
    "original_deadline_epoch": END,
    "finished_epoch": time.time(),
    "registration_sha256": sha(REGPATH),
    "controller_sha256": sha(__file__),
    "GPU_used": False,
    "strength_success": False,
}
if status == "failed-preserved":
    result["error"] = error
(BASE / "cohort-result.json").open("x").write(json.dumps(result, indent=2) + "\n")
print(status, flush=True)
