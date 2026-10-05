"""Root owner for frozen admission plus classical paired strength development games."""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

regpath = Path(sys.argv[1])
reg = json.loads(regpath.read_text())
base = regpath.parent
ram = Path(reg["ram_stage"])
end = reg["deadline_epoch"]
first = reg["first_epoch"]
child = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


assert first <= time.time() < end <= 1791273600
assert end - first == 7200
assert sha(__file__) == reg["controller_sha256"]
os.sched_setaffinity(0, {reg["cpu_core"]})
ram.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location("frozen_budget", reg["memory_guard_path"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
budget = module.CgroupMemoryBudget(16106127360)
env = {
    **os.environ,
    "PYTHONPATH": reg["checkout"] + "/src",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def guard(deadline=end):
    if time.time() >= min(end, deadline):
        raise TimeoutError("original registered arena/profile clock exhausted")
    if shutil.disk_usage("/workspace").free < 268435456:
        raise RuntimeError("workspace floor breached")
    budget.check()
    if sum(p.stat().st_size for p in ram.rglob("*") if p.is_file()) > 100663296:
        raise RuntimeError("registered RAM stage96MiB cap exceeded")
    for path, expected in reg["input_pins"].items():
        if sha(path) != expected:
            raise ValueError("frozen helper/input changed: " + path)


def invoke(command, label, deadline):
    global child
    guard(deadline)
    with (
        (ram / (label + ".stdout.log")).open("x") as out,
        (ram / (label + ".stderr.log")).open("x") as err,
    ):
        child = subprocess.Popen(
            command, cwd=reg["checkout"], env=env, stdout=out, stderr=err, start_new_session=True
        )
        while child.poll() is None:
            guard(deadline)
            time.sleep(0.5)
        code = child.returncode
        child = None
    guard(deadline)
    if code:
        raise RuntimeError(label + " failed, exit " + str(code))


rows = []
try:
    invoke(reg["profile_command"], "profile", reg["profile_deadline_epoch"])
    qpath = Path(reg["qualification_path"])
    qualification = json.loads(qpath.read_text())
    assert qualification["status"] == "PASS-classical-mixed-search-qualification-not-strength"
    assert len(qualification["rows"]) == 24
    expected = {
        (seed, role, opening)
        for seed in (20262905, 20262906)
        for role in ("e8", "prior", "learned")
        for opening in range(4)
    }
    assert {(row["seed"], row["role"], row["opening"]) for row in qualification["rows"]} == expected
    assert qualification["protocol_sha256"] == sha(reg["protocol_path"])
    rows.append(
        {
            "label": "profile",
            "status": qualification["status"],
            "sha256": sha(qpath),
            "finished_epoch": time.time(),
        }
    )
    publish(base / "profile-owner-result.json", rows[-1])
    invoke(reg["arena_command"], "arena", end)
    cohort_path = Path(reg["arena_output"]) / "cohort-result.json"
    cohort = json.loads(cohort_path.read_text())
    assert cohort["status"] == "completed-games-not-strength", cohort["status"]
    rows.append(
        {
            "label": "arena",
            "status": cohort["status"],
            "sha256": sha(cohort_path),
            "finished_epoch": time.time(),
        }
    )
    guard()
    status = "completed-160-games-awaiting-independent-strength-audit"
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
    "original_first_epoch": first,
    "original_deadline_epoch": end,
    "finished_epoch": time.time(),
    "registration_sha256": sha(regpath),
    "controller_sha256": sha(__file__),
    "GPU_used": False,
    "strength_success_claimed": False,
}
if status == "failed-preserved":
    result["error"] = error
publish(base / "owner-result.json", result)
print(json.dumps(result), flush=True)
