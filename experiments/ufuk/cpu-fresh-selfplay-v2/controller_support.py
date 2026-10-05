"""Frozen real-actor wiring and guarded owned CPU child; no implicit execution."""

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SOURCE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
E8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
END = 1791273600.0
HELPERS = {
    "journal_v1.py",
    "journal_v2.py",
    "produce_v2.py",
    "anchor_value.py",
    "search.py",
    "value.py",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError("immutable input SHA differs")
    return json.loads(Path(path).read_text())


def validate_clock(first, deadline, seconds, now):
    if first > now or deadline != first + seconds or now >= deadline or deadline > END:
        raise ValueError("original first/deadline invalid or exhausted; no reset")


def bindings(manifest_path, manifest_sha, maximum):
    manifest = read_bound(manifest_path, manifest_sha)
    if manifest["schema"] != "fresh-qsearch-realCPU-controller-manifest-v1":
        raise ValueError("new actor manifest required")
    config = read_bound(manifest["config"]["path"], manifest["config"]["sha256"])
    if config["source_commit"] != SOURCE or manifest["source_commit"] != SOURCE:
        raise ValueError("exact clean6fcc source required")
    if config["max_actions"] != maximum or config["actors"] != 1:
        raise ValueError("fixed actor/action ceiling differs")
    if config["model_sha256"] != E8 or config["anchor_model_sha256"] != E8:
        raise ValueError("real frozen e8 snapshot and anchor required")
    if config["evaluator_identity"] != "frozen-real-NeuralValue-CPU":
        raise ValueError("synthetic evaluator is not qualification")
    if set(manifest["helpers"]) != HELPERS:
        raise ValueError("complete transitive actor helper closure required")
    for key, expected in [("model", E8), ("anchor_model", E8)]:
        row = manifest[key]
        if row["sha256"] != expected or sha(row["path"]) != expected:
            raise ValueError("model/anchor SHA differs")
    for row in manifest["helpers"].values():
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("helper SHA differs")
    keys = {
        "journal_v2.py": "producer_sha256",
        "produce_v2.py": "runner_sha256",
        "anchor_value.py": "anchor_helper_sha256",
        "value.py": "value_helper_sha256",
        "search.py": "search_helper_sha256",
    }
    for name, config_key in keys.items():
        if manifest["helpers"][name]["sha256"] != config[config_key]:
            raise ValueError("config/helper binding differs")
    producer_dir = Path(manifest["helpers"]["produce_v2.py"]["path"]).resolve().parent
    for name in ("journal_v1.py", "journal_v2.py"):
        if Path(manifest["helpers"][name]["path"]).resolve() != producer_dir / name:
            raise ValueError("actual sibling import path differs")
    return manifest, config


def command(manifest, config, target, output, parent=None, parent_sha=None):
    if not 0 < target <= config["max_actions"]:
        raise ValueError("target exceeds fixed epoch")
    if (parent is None) != (parent_sha is None):
        raise ValueError("parent path/SHA must be supplied together")
    if parent is not None and sha(parent) != parent_sha:
        raise ValueError("immutable parent checkpoint SHA differs")
    helper = manifest["helpers"]
    args = [
        sys.executable,
        helper["produce_v2.py"]["path"],
        "--config",
        manifest["config"]["path"],
        "--config-sha256",
        manifest["config"]["sha256"],
        "--source-repo",
        manifest["checkout"],
        "--model",
        manifest["model"]["path"],
        "--anchor-model",
        manifest["anchor_model"]["path"],
        "--search-helper",
        helper["search.py"]["path"],
        "--value-helper",
        helper["value.py"]["path"],
        "--anchor-helper",
        helper["anchor_value.py"]["path"],
        "--output",
        str(output),
        "--target-actions",
        str(target),
        "--original-deadline",
        str(config["original_deadline_epoch"]),
    ]
    if parent is not None:
        args += ["--resume", str(parent), "--resume-sha256", parent_sha]
    return args


def runtime(manifest, workspace, ram_stage):
    core = manifest["cpu_core"]
    if type(core) is not int or core not in os.sched_getaffinity(0):
        raise ValueError("explicit registered CPU core is outside allowed affinity")
    os.sched_setaffinity(0, {core})
    checkout = Path(manifest["checkout"])
    owned = load(checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py", "fresh_actor_owned")
    owned.check_source(checkout, SOURCE)
    sys.path.insert(0, str(checkout / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 1024**3)
    ram_stage = Path(ram_stage).resolve()
    if Path("/dev/shm") not in ram_stage.parents:
        raise ValueError("actor artifact stage must be under /dev/shm")

    def guard(deadline):
        if time.time() >= min(deadline, END):
            raise TimeoutError("original actor deadline exhausted")
        memory = budget.check()
        free = shutil.disk_usage(workspace).free
        total = (
            sum(p.stat().st_size for p in ram_stage.rglob("*") if p.is_file())
            if ram_stage.exists()
            else 0
        )
        if free < 256 * 1024**2 or total > 16 * 1024**2:
            error = RuntimeError("registered disk/RAM-stage guard failed")
            error.snapshot = {
                "workspace_free_bytes": free,
                "ram_stage_bytes": total,
                "memory": memory,
            }
            raise error
        return {"workspace_free_bytes": free, "ram_stage_bytes": total, "memory": memory}

    return owned, guard


def guarded_child(args, manifest, deadline, guard, logdir, label, publish):
    """Only the child started here is ever signalled; no name-based process kill."""
    guard(deadline)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(manifest["checkout"]) / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    started = time.time()
    receipt = {"command": args, "started_epoch": started, "original_deadline_epoch": deadline}
    with (
        (logdir / (label + ".stdout.log")).open("xb") as out,
        (logdir / (label + ".stderr.log")).open("xb") as err,
    ):
        process = subprocess.Popen(
            args, cwd=manifest["checkout"], env=env, start_new_session=True, stdout=out, stderr=err
        )
        receipt["pid"] = process.pid
        try:
            receipt["startticks"] = int(
                Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()[19]
            )
            publish(logdir / (label + ".command.json"), receipt)
            while process.poll() is None:
                guard(deadline)
                time.sleep(0.2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, args)
            guard(deadline)
            receipt["status"] = "completed-new-actor-segment-not-strength"
        except BaseException as exc:
            receipt.update(
                status="failed-preserved",
                error=repr(exc),
                failure_snapshot=getattr(exc, "snapshot", None),
            )
            raise
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            receipt.update(finished_epoch=time.time(), returncode=process.returncode)
            publish(logdir / (label + ".process-result.json"), receipt)
    return receipt


def journal_module(manifest):
    directory = Path(manifest["helpers"]["journal_v2.py"]["path"]).parent
    sys.path.insert(0, str(directory))
    return load(directory / "journal_v2.py", "fresh_actor_journal_v2")
