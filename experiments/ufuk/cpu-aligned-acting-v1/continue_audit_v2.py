"""Resume only the failed auditor stage; no retraining and no audit-clock reset."""

import argparse
import concurrent.futures
import importlib.util
import json
import os
import signal
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "study", "output", "weights", "stockfish"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    protocol = json.loads((args.study / "protocol.json").read_text())
    clocks = json.loads((args.output / "qualification-owner-process.json").read_text())
    owned.check_source(args.checkout, protocol["source_commit"])
    environment = {
        **os.environ,
        "PYTHONPATH": str(args.checkout / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    files = [
        args.study / name
        for name in (
            "protocol.json",
            "run.py",
            "audit_v2.py",
            "arena-protocol.json",
            "arena.py",
            "audit_arena.py",
        )
    ]
    hashes = {str(p): owned.sha(p) for p in files}
    owned.publish(
        args.output / "pipeline-v2-contract.json",
        {
            "helper_sha256": owned.sha(Path(__file__)),
            "inputs": hashes,
            "source_commit": protocol["source_commit"],
            "GPU_used": False,
            "qualification_original_first_epoch": clocks["original_first_epoch"],
            "qualification_original_deadline_epoch": clocks["original_deadline_epoch"],
        },
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"CPU pipeline interrupted {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def execute(tag, command, seconds, absolute=None):
        assert all(owned.sha(Path(p)) == h for p, h in hashes.items())
        first = time.time()
        deadline = min(first + seconds, absolute) if absolute else first + seconds
        tag += "-v2"
        row = {
            "command": command,
            "original_first_epoch": first,
            "original_deadline_epoch": deadline,
            "GPU_used": False,
        }
        owned.publish(args.output / (tag + ".pipeline-invocation.json"), row)
        try:
            with (
                (args.output / (tag + ".pipeline-stdout.log")).open("xb") as out,
                (args.output / (tag + ".pipeline-stderr.log")).open("xb") as err,
            ):
                owned.run_owned(
                    command,
                    cwd=args.checkout,
                    env=environment,
                    stdout=out,
                    stderr=err,
                    deadline=deadline,
                )
            row["status"] = "completed-awaiting-result"
        except BaseException as exc:
            row.update(status="failed-preserved", error=repr(exc))
            raise
        finally:
            row["finished_epoch"] = time.time()
            owned.publish(args.output / (tag + ".pipeline-result.json"), row)

    try:
        previous = json.loads((args.output / "pipeline-contract.json").read_text())
        assert all(owned.sha(Path(p)) == h for p, h in previous["inputs"].items())
        assert (
            json.loads((args.output / "pipeline-result.json").read_text())["status"]
            == "failed-preserved"
        )
        fits = args.output / "fits"
        assert (
            json.loads((fits / "cohort-result.json").read_text())["status"]
            == "completed-runs-not-strength"
        )

        def audit(tag):
            command = [
                os.sys.executable,
                str(args.study / "audit_v2.py"),
                "--run",
                str(fits / tag),
                "--config",
                str(fits / (tag + ".config.json")),
                "--protocol",
                str(args.study / "protocol.json"),
                "--weights",
                str(args.weights),
                "--output",
                str(fits / (tag + ".fresh-audit.json")),
            ]
            old = json.loads((args.output / (tag + "-audit.pipeline-invocation.json")).read_text())
            execute(tag + "-audit", command, 900, old["original_deadline_epoch"])
            receipt = json.loads((fits / (tag + ".fresh-audit.json")).read_text())
            assert receipt["status"] == "pass-data-actor-RNG-certificates-native-not-strength"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(audit, protocol["configs"]))
        first = time.time()
        arena_protocol = json.loads((args.study / "arena-protocol.json").read_text())
        command = [
            os.sys.executable,
            str(args.study / "arena.py"),
            "--checkout",
            str(args.checkout),
            "--protocol",
            str(args.study / "arena-protocol.json"),
            "--weights",
            str(args.weights),
            "--stockfish",
            str(args.stockfish),
            "--fits",
            str(fits),
            "--output",
            str(args.output / "arena"),
            "--first-epoch",
            str(first),
        ]
        execute(
            "arena",
            command,
            arena_protocol["whole_seconds"],
            first + arena_protocol["whole_seconds"],
        )
        assert (
            json.loads((args.output / "arena/cohort-result.json").read_text())["status"]
            == "completed-games-not-strength"
        )
        command = [
            os.sys.executable,
            str(args.study / "audit_arena.py"),
            "--protocol",
            str(args.study / "arena-protocol.json"),
            "--arena",
            str(args.output / "arena"),
            "--fits",
            str(fits),
            "--output",
            str(args.output / "arena/independent-fullhistory-result.json"),
        ]
        execute("fullhistory", command, 600)
        result = json.loads((args.output / "arena/independent-fullhistory-result.json").read_text())
        owned.publish(
            args.output / "pipeline-v2-result.json",
            {
                "status": "completed-controls-not-formal-strength",
                "screen_result": result["screen_result"],
                "games": result["games"],
                "strength_success_claimed": False,
                "GPU_used": False,
                "finished_epoch": time.time(),
            },
        )
    except BaseException as exc:
        owned.publish(
            args.output / "pipeline-v2-result.json",
            {
                "status": "failed-preserved",
                "error": repr(exc),
                "finished_epoch": time.time(),
                "strength_success_claimed": False,
                "GPU_used": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
