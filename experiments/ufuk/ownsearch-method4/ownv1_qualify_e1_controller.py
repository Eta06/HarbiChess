"""Owned single900s read-only E1 auditor qualification; no retry or profile clock reset."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from ownv1_audit_support import check_source, guard, publish, sha


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    for name in ("manifest", "root", "python", "helper"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--helper-sha256", required=True)
    a = p.parse_args()
    assert sha(a.manifest) == a.manifest_sha256 and sha(a.helper) == a.helper_sha256
    spec = json.loads(a.manifest.read_text())
    assert spec["scope"] == "existing-development-E1-not-formal-method4"
    assert spec["helper_sha256"][a.helper.name] == a.helper_sha256
    assert spec["controller_sha256"] == sha(__file__)
    deadline = spec["started_epoch"] + spec["whole_seconds"]
    assert 0 < spec["whole_seconds"] <= 900
    assert spec["started_epoch"] <= time.time() < deadline < 1791170400
    repo = Path(spec["checkout"])
    check_source(repo, spec["source_commit"])
    a.root.mkdir(parents=True, exist_ok=False)
    process = None
    status, error = "failed-preserved-no-retry-E1-qualification", None
    command = [
        str(a.python),
        str(a.helper.resolve()),
        "--manifest",
        str(a.manifest.resolve()),
        "--manifest-sha256",
        a.manifest_sha256,
        "--output",
        str((a.root / "qualification.json").resolve()),
        "--deadline-epoch",
        str(deadline),
    ]
    publish(
        a.root / "command.json",
        {
            "argv": command,
            "original_qualification_firstclock": spec["started_epoch"],
            "absolute_deadline_epoch": deadline,
            "manifest_sha256": a.manifest_sha256,
            "original_profile_deadline_unchanged": spec["original_profile_deadline_epoch"],
        },
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned E1 qualifier interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Existing development data may still publish; wait consumes SAME900 qualification clock.
        run = Path(spec["run"])
        while not (run / "checkpoints/epoch-00000001/checkpoint.json").is_file():
            guard(deadline, a.root)
            time.sleep(0.5)
        while True:
            try:
                development = json.loads(Path(spec["development_controller_result"]).read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                guard(deadline, a.root)
                time.sleep(0.5)
        assert development["status"] == "pass-one-development-epoch-and-readonly-audit"
        assert development["source_commit"] == spec["source_commit"]
        assert development["finished_epoch"] <= spec["original_profile_deadline_epoch"]
        assert development["absolute_deadline_epoch"] == spec["original_profile_deadline_epoch"]
        env = {
            **os.environ,
            "PYTHONPATH": str(repo.resolve() / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        with (a.root / "stdout.log").open("x") as out, (a.root / "stderr.log").open("x") as err:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            while process.poll() is None:
                guard(deadline, a.root)
                time.sleep(0.5)
        assert process.returncode == 0
        result = json.loads((a.root / "qualification.json").read_text())
        assert (
            result["status"]
            == "pass-actualCUDA-E1-full-data-original-groups-raw-packets-and-targeted-mutations"
        )
        assert result["source_commit"] == spec["source_commit"]
        assert result["manifest_sha256"] == a.manifest_sha256
        assert result["audit_report"]["raw_actor_replayed"] == 32768
        assert result["audit_report"]["raw_actor_packet_roots_verified"] == 18
        assert set(result["targeted_actual_data_mutations_rejected"]) == {
            "raw_base_policy",
            "terminal_WDL",
            "search_policy",
            "frozen_storage_onebit",
        }
        assert all(result["targeted_actual_data_mutations_rejected"].values())
        assert result["optimizer_updates_performed_by_qualification"] == 0
        assert result["new_selfplay_transitions_generated"] == 0
        guard(deadline, a.root)
        status = "completed-actualCUDA-E1-auditor-development-qualification-only"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if process is not None:
            stop(process)
        publish(
            a.root / "result.json",
            {
                "status": status,
                "error": error,
                "returncode": process.returncode if process else None,
                "original_qualification_started_epoch": spec["started_epoch"],
                "absolute_deadline_epoch": deadline,
                "finished_epoch": time.time(),
                "source_commit": spec["source_commit"],
                "controller_sha256": sha(__file__),
                "manifest_sha256": a.manifest_sha256,
                "qualification_sha256": sha(a.root / "qualification.json")
                if status.startswith("completed")
                else None,
                "scope": (
                    "E1 DEVELOPMENT ONLY; does not claim E+1 formal audit or strength qualification"
                ),
            },
        )


if __name__ == "__main__":
    main()
