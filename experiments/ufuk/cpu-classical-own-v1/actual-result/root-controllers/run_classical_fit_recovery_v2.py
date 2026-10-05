"""Root recovery fixes output-parent scope only; original fit clock/inputs unchanged."""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

BASE = Path("/workspace/work/harbichess/cpu-classical-own-v1-actual/offline-fit-recovery-v2")
REGPATH = BASE / "registration.json"
REG = json.loads(REGPATH.read_text())
END = REG["original_deadline_epoch"]
RAM = Path(REG["ram_stage"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, v):
    with p.open("x") as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


assert REG["original_first_epoch"] <= time.time() < END <= 1791273600
assert sha(__file__) == REG["controller_sha256"]
for p, expected in REG["input_pins"].items():
    assert sha(p) == expected, p
spec = importlib.util.spec_from_file_location("budget", REG["memory_guard_path"])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
budget = m.CgroupMemoryBudget(16106127360)
RAM.mkdir(exist_ok=False)
child = None
rows = []
env = {
    **os.environ,
    "PYTHONPATH": REG["checkout"] + "/src",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def guard():
    if time.time() >= END:
        raise TimeoutError("original1800fitdeadline exhausted")
    budget.check()
    if shutil.disk_usage("/workspace").free < 268435456:
        raise RuntimeError("workspace256MiBfloor")
    if sum(p.stat().st_size for p in RAM.rglob("*") if p.is_file()) > 33554432:
        raise RuntimeError("two scoped16MiB output roots exceeded")
    for p, expected in REG["input_pins"].items():
        if sha(p) != expected:
            raise ValueError("immutable input/source changed: " + p)


def invoke(cmd, name):
    global child
    guard()
    with (
        (BASE / (name + ".stdout.log")).open("x") as out,
        (BASE / (name + ".stderr.log")).open("x") as err,
    ):
        child = subprocess.Popen(
            cmd, cwd=REG["checkout"], env=env, stdout=out, stderr=err, start_new_session=True
        )
        while child.poll() is None:
            guard()
            time.sleep(0.5)
        rc = child.returncode
        child = None
    guard()
    if rc:
        raise RuntimeError(name + " exited " + str(rc))


try:
    for job in REG["jobs"]:
        seed = job["seed"]
        output = Path(job["output"])
        output.parent.mkdir(exist_ok=False)
        invoke(job["command"], str(seed) + "-fit")
        text = (BASE / (str(seed) + "-fit.stdout.log")).read_text()
        result = json.loads(text.strip().splitlines()[-1])
        assert result["status"] == "completed-offline-fit-not-strength"
        assert result["step"] == job["expected_updates"]
        initial = output / "step-00000000.json"
        final = output / f"step-{job['expected_updates']:08d}.json"
        for step, native in ((0, initial), (job["expected_updates"], final)):
            n = json.loads(native.read_text())
            assert n["step"] == step
            assert n["contract"]["dataset_sha256"] == job["dataset_sha256"]
            assert n["contract"]["original_deadline"] == END
            invoke(
                [
                    *job["command"],
                    "--resume",
                    str(native),
                    "--resume-sha256",
                    sha(native),
                    "--audit-only",
                ],
                str(seed) + f"-strict-native-{step}",
            )
        candidate = output / "candidate.json"
        row = {
            "seed": seed,
            "status": "PASS-fixed-final-fit-and-two-fresh-native-loads-not-strength",
            "finished_observed_epoch": time.time(),
            "updates": job["expected_updates"],
            "candidate_path": str(candidate),
            "candidate_sha256": sha(candidate),
            "initial_native_path": str(initial),
            "initial_native_sha256": sha(initial),
            "final_native_path": str(final),
            "final_native_sha256": sha(final),
            "dataset_sha256": job["dataset_sha256"],
            "GPU_used": False,
            "teacher_labels": False,
        }
        guard()
        publish(BASE / (str(seed) + "-fit-result.json"), row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    guard()
    status = "PASS-both-final-fits-and-four-fresh-native-loads-not-strength"
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
r = {
    "status": status,
    "rows": rows,
    "original_first_epoch": REG["original_first_epoch"],
    "original_deadline_epoch": END,
    "finished_epoch": time.time(),
    "registration_sha256": sha(REGPATH),
    "controller_sha256": sha(__file__),
    "strength_success_claimed": False,
}
if status == "failed-preserved":
    r["error"] = error
publish(BASE / "cohort-result.json", r)
print(status, flush=True)
