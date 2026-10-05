"""Owned CPU aligned acting controls, admitted only after real CLI and disk preflight."""

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "protocol", "weights", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    protocol = json.loads(args.protocol.read_text())
    deadline = args.first_epoch + protocol["fit_whole_seconds"]
    owned.check_source(args.checkout, protocol["source_commit"])
    if args.first_epoch > time.time() or time.time() >= deadline:
        raise ValueError("observed original finite campaign clock required")
    args.output.mkdir(exist_ok=False)
    contract = {
        "protocol_sha256": owned.sha(args.protocol),
        "runner_sha256": owned.sha(Path(__file__)),
        "source_commit": protocol["source_commit"],
        "original_first_epoch": args.first_epoch,
        "original_whole_deadline_epoch": deadline,
        "GPU_used": False,
    }
    owned.publish(args.output / "owner-contract.json", contract)
    book = Path(protocol["book"]["path"])
    if owned.sha(book) != protocol["book"]["sha256"] or (
        owned.sha(args.weights) != protocol["initial_e8_sha256"]
    ):
        raise ValueError("initial model/book binding differs")
    proof = json.loads((args.qualification / "result.json").read_text())
    if proof.get("status") != "pass":
        raise ValueError("actual all-search CLI qualification must PASS")
    journals = list((args.qualification / "whole/journal").glob("*.json.gz"))
    assert len(journals) == 2
    # Conservative measured projection, not a storage guarantee. Guards remain hard.
    max_tiny_journal = max(p.stat().st_size for p in journals)
    jobs = len(protocol["configs"])
    tiny = json.loads((args.qualification / "whole/metadata.json").read_text())["config"]
    tiny_actions = tiny["actors"]["games"] * tiny["epoch_steps"]
    full = next(iter(protocol["configs"].values()))
    full_actions = full["actors"]["games"] * full["epoch_steps"]
    projected = jobs * (
        max_tiny_journal * full_actions / tiny_actions * (protocol["epochs"] + 1) + 8 * 1024**2
    )
    free = shutil.disk_usage(args.output).free
    required = protocol["disk_min_free_bytes"] + 1.5 * projected
    admission = {
        "max_observed_tiny_journal_bytes": max_tiny_journal,
        "tiny_actions": tiny_actions,
        "full_actions": full_actions,
        "projected_output_bytes": projected,
        "reserved_extra_native_bytes_per_job": 8 * 1024**2,
        "free_bytes": free,
        "required_bytes": required,
        "safety_factor": 1.5,
        "admitted": free >= required,
        "qualification_result_sha256": owned.sha(args.qualification / "result.json"),
    }
    owned.publish(args.output / "resource-admission.json", admission)
    if not admission["admitted"]:
        owned.publish(
            args.output / "cohort-result.json", {"status": "not-admitted-disk", **admission}
        )
        return

    def run(tag, config):
        first = time.time()
        end = min(first + protocol["fit_original_seconds_per_job"], deadline)
        directory = args.output / tag
        config_path = args.output / (tag + ".config.json")
        owned.publish(config_path, config)
        command = [
            os.sys.executable,
            "-m",
            "harbichess.training.torch_search_acting_run",
            str(directory),
            "--weights",
            str(args.weights),
            "--book",
            str(book),
            "--config",
            str(config_path),
            "--protocol",
            str(args.protocol),
            "--source-commit",
            protocol["source_commit"],
            "--max-epochs",
            str(protocol["epochs"]),
            "--checkpoint-interval",
            str(protocol["checkpoint_interval"]),
            "--deadline-epoch",
            str(end),
            "--memory-max-bytes",
            str(protocol["memory_max_bytes"]),
            "--memory-policy",
            protocol["memory_policy"],
            "--disk-min-free-bytes",
            str(protocol["disk_min_free_bytes"]),
        ]
        row = {
            "tag": tag,
            "first_epoch": first,
            "original_deadline_epoch": end,
            "command": command,
            "config_sha256": owned.sha(config_path),
            "scope": "CPU own-selfplay development; no strength or originality claim",
        }
        owned.publish(args.output / (tag + ".invocation.json"), row)
        try:
            if owned.sha(args.protocol) != contract["protocol_sha256"]:
                raise ValueError("frozen protocol changed")
            with (
                (args.output / (tag + ".stdout.log")).open("xb") as out,
                (args.output / (tag + ".stderr.log")).open("xb") as err,
            ):
                owned.run_owned(
                    command,
                    cwd=args.checkout,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(args.checkout / "src"),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                    stdout=out,
                    stderr=err,
                    deadline=end,
                )
            result = json.loads((directory / "progress.json").read_text())
            assert result["status"] == "completed" and result["epoch"] == protocol["epochs"]
            row.update(status="completed-awaiting-independent-audit-and-games", receipt=result)
        except BaseException as exc:
            row.update(status="failed-preserved", error=repr(exc))
        row["finished_epoch"] = time.time()
        owned.publish(args.output / (tag + ".owner-result.json"), row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ("command", "receipt")}),
            flush=True,
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=protocol["max_concurrent_jobs"]) as pool:
        futures = [pool.submit(run, tag, config) for tag, config in protocol["configs"].items()]
        rows = [future.result() for future in futures]
    owned.publish(
        args.output / "cohort-result.json",
        {
            "status": "completed-runs-not-strength"
            if all(r["status"].startswith("completed-") for r in rows)
            else "failed-preserved",
            "rows": rows,
            "contract": contract,
            "finished_epoch": time.time(),
        },
    )


if __name__ == "__main__":
    main()
