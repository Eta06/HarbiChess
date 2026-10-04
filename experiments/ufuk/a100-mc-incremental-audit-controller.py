"""Owned whole21000seconds read-only incremental audit; no retry/resume mode."""

import argparse
import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "whole_controller",
    Path(__file__).with_name("a100-mc-whole-training-controller-v3-18000.py"),
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "python", "manifest", "run", "audit-script", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--audit-script-sha256", required=True)
    a = p.parse_args()
    assert c.sha(a.manifest) == a.manifest_sha256 and c.sha(a.audit_script) == a.audit_script_sha256
    m = c.read(a.manifest)
    source = m["source_commit"]
    started = m["original_training_started_epoch"]
    trainingdeadline = m["original_training_deadline_epoch"]
    assert trainingdeadline == started + 18000
    deadline = min(started + 21000, c.HARD_DEADLINE)
    assert started <= time.time() < deadline
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo, text=True).strip()
        == source
        and not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=a.repo, text=True
        ).strip()
    )
    a.root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "method2-incremental21000audit-controller-v1",
        "source_commit": source,
        "manifest_sha256": a.manifest_sha256,
        "audit_script_sha256": a.audit_script_sha256,
        "controller_sha256": c.sha(Path(__file__)),
        "original_training_started_epoch": started,
        "original_training_deadline_epoch": trainingdeadline,
        "original_audit_deadline_epoch": deadline,
        "registered_audit_wholemaximum_seconds": 21000,
        "memory_max_bytes": 64 * 1024**3,
        "disk_min_free_bytes": 8 * 1024**3,
        "scope": (
            "Read-only immutable all41native/all40data audits; includes waitin"
            "g;0optimizerupdates; no per-epoch reset/resume/skip"
        ),
        "status": "running",
    }
    args = [
        str(a.python),
        str(a.audit_script.resolve()),
        "--manifest",
        str(a.manifest.resolve()),
        "--run",
        str(a.run.resolve()),
        "--output",
        str((a.root / "full-audit-result.json").resolve()),
        "--deadline-epoch",
        str(deadline),
    ]
    receipt["command"] = args
    c.publish(a.root / "command.json", receipt)
    env = {
        **os.environ,
        "PYTHONPATH": str(a.repo.resolve() / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    process = None
    c.install_interrupt_handlers()
    try:
        with (
            (a.root / "stdout.log").open("x") as out,
            (a.root / "stderr.log").open("x") as err,
        ):
            process = subprocess.Popen(
                args,
                cwd=a.repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            c.publish(
                a.root / "owner.json",
                {
                    "pid": process.pid,
                    "owned_process_group": process.pid,
                    "original_audit_deadline_epoch": deadline,
                },
            )
            while process.poll() is None:
                memory, _ = c.cpu_total_memory()
                if (
                    time.time() >= deadline
                    or memory > 64 * 1024**3
                    or __import__("shutil").disk_usage(a.root).free < 8 * 1024**3
                ):
                    c.terminate(process)
                    raise TimeoutError("Originalwhole21000audit/resourceceilings exhausted")
                time.sleep(0.5)
        assert process.returncode == 0 and time.time() <= deadline
        final = c.read(a.root / "full-audit-result.json")
        assert (
            final["status"] == "pass-all41native-all40epochs-independent-own-data"
            and final["audited_native_checkpoints"] == 41
            and final["independently_replayed_fresh_transitions"] == 1310720
            and final["optimizer_updates_performed_by_this_audit"] == 0
        )
        receipts = final["immutable_percheckpoint_audit_receipt_sha256"]
        assert len(receipts) == 41
        for path, digest in receipts.items():
            assert c.sha(path) == digest
        receipt.update(
            status="completed-independent-all41native-all40own-data-audit",
            full_audit_sha256=c.sha(a.root / "full-audit-result.json"),
        )
    except BaseException as e:
        if process is not None and process.poll() is None:
            c.terminate(process)
        receipt.update(
            status="failed-or-incomplete-incremental-audit-preserved",
            error=repr(e),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        receipt.update(
            finished_epoch=time.time(),
            whole_since_original_training_start=time.time() - started,
            returncode=process.returncode if process else None,
        )
        c.publish(a.root / "result.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
