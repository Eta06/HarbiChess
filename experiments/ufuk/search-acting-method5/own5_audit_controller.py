"""Read-only SEARCH-ACTING-v2 audit from original first clock; no budget reset."""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from own5_audit_support import check_source, guard, publish, sha
from own5_training_controller import read, stop

END = 1791170400


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "repo", "python", "audit-script", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit-script-sha256", required=True)
    args = parser.parse_args()
    assert sha(args.manifest) == args.manifest_sha256
    assert sha(args.audit_script) == args.audit_script_sha256
    spec = read(args.manifest)
    assert spec["qualification_ledger_slot"] == 5
    first = spec["original_training_started_epoch"]
    assert (
        spec["original_training_deadline_epoch"]
        == first + spec["whole_training_seconds"]
    )
    deadline = min(first + spec["whole_audit_seconds"], END)
    assert first <= time.time() < deadline
    assert Path(spec["producer_checkout"]).resolve() == args.repo.resolve()
    check_source(args.repo, spec["source_commit"])
    args.root.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.python),
        str(args.audit_script.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--manifest-sha256",
        args.manifest_sha256,
        "--run",
        spec["run"],
        "--output",
        str((args.root / "full-audit-result.json").resolve()),
        "--deadline-epoch",
        str(deadline),
    ]
    publish(
        args.root / "command.json",
        {
            "argv": command,
            "original_first_clock": first,
            "absolute_deadline_epoch": deadline,
            "manifest_sha256": args.manifest_sha256,
            "audit_script_sha256": args.audit_script_sha256,
            "controller_sha256": sha(Path(__file__)),
            "source_commit": spec["source_commit"],
        },
    )

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Owned audit interruption {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    process = None
    status = "failed-or-incomplete-own5-audit-preserved"
    error = None
    try:
        env = {
            **os.environ,
            "PYTHONPATH": str(args.repo.resolve() / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        with (
            (args.root / "stdout.log").open("x") as out,
            (args.root / "stderr.log").open("x") as err,
        ):
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            while process.poll() is None:
                guard(deadline, args.root)
                time.sleep(0.5)
        assert process.returncode == 0
        result = read(args.root / "full-audit-result.json")
        assert (
            result["status"]
            == "pass-all-fixed-search-acting-v2-native-own-data-search-ledgers"
        )
        assert result["audited_native_checkpoints"] == spec["fixed_epochs"] + 1
        assert (
            result["independently_replayed_fresh_transitions"]
            == spec["fixed_epochs"] * 32768
        )
        assert result["prescribed_neural_roots_verified"] == 108
        assert result["raw_actor_packet_roots_verified"] == 108
        assert result["neural_search_roots_recomputed"] >= 108
        assert result["optimizer_updates_performed_by_this_audit"] == 0
        assert result["manifest_sha256"] == args.manifest_sha256
        assert result["audit_sha256"] == args.audit_script_sha256
        assert (
            len(result["immutable_percheckpoint_audit_receipt_sha256"])
            == spec["fixed_epochs"] + 1
        )
        for path, digest in result[
            "immutable_percheckpoint_audit_receipt_sha256"
        ].items():
            assert sha(path) == digest
        guard(deadline, args.root)
        status = "completed-independent-search-acting-v2-allnative-data-search-ledgers"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if process is not None:
            stop(process)
        publish(
            args.root / "result.json",
            {
                "status": status,
                "error": error,
                "source_commit": spec["source_commit"],
                "original_first_clock": first,
                "absolute_deadline_epoch": deadline,
                "finished_epoch": time.time(),
                "returncode": process.returncode if process else None,
                "full_audit_sha256": sha(args.root / "full-audit-result.json")
                if status.startswith("completed")
                else None,
            },
        )


if __name__ == "__main__":
    main()
