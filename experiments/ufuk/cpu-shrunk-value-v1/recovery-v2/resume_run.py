"""One owned CPU arena worker, immutable models and prior actual full-native receipts."""

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "study", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--parent-arena", type=Path, required=True)
    parser.add_argument("--parent-contract-sha256", required=True)
    parser.add_argument("--recovery-helper-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    p = json.loads((args.study / "protocol.json").read_text())
    owned.check_source(args.checkout, p["source_commit"])
    q = json.loads(args.qualification.read_text())
    assert q["status"] == "PASS-qualification-not-strength"
    assert q["protocol_sha256"] == owned.sha(args.study / "protocol.json") and len(q["rows"]) == 24
    assert args.first_epoch <= time.time()
    end = args.first_epoch + p["whole_seconds"]
    assert end > time.time()
    parent_contract_path = args.parent_arena / "owner-contract.json"
    assert owned.sha(parent_contract_path) == args.parent_contract_sha256
    parent_contract = json.loads(parent_contract_path.read_text())
    assert parent_contract["original_first_epoch"] == args.first_epoch
    assert parent_contract["original_deadline_epoch"] == end
    args.output.mkdir(exist_ok=False)
    hashes = {
        name: owned.sha(args.study / name)
        for name in ("protocol.json", "tournament.py", "search.py", "value.py")
    }
    assert parent_contract["source_commit"] == p["source_commit"]
    assert parent_contract["qualification_sha256"] == owned.sha(args.qualification)
    assert parent_contract["inputs"] == hashes
    contract = {
        **parent_contract,
        "recovery_executor_sha256": owned.sha(args.recovery_helper_dir / "resume_tournament.py"),
        "recovery_runner_sha256": owned.sha(Path(__file__)),
        "recovery_support_sha256": owned.sha(args.recovery_helper_dir / "recovery_support.py"),
        "parent_owner_contract_sha256": args.parent_contract_sha256,
        "parent_owner_contract_path": str(parent_contract_path),
    }
    owned.publish(args.output / "owner-contract.json", contract)
    free = shutil.disk_usage(args.output).free
    required = p["disk_min_free_bytes"] + 1.5 * p["reserved_output_bytes"]
    owned.publish(
        args.output / "resource-admission.json",
        {"free_bytes": free, "required_bytes": required, "admitted": free >= required},
    )
    if free < required:
        raise RuntimeError("registered no-deletion disk admission failed")
    # Existing immutable offline-native manifests/payload SHA, no old training/resume clock reset.
    for seed in p["match_seeds"]:
        for role, r in p["models"][str(seed)].items():
            weight = Path(r["path"])
            assert owned.sha(weight) == r["sha256"]
            if role != "e8":
                manifest = weight.parent / "checkpoint.json"
                assert owned.sha(manifest) == r["native_manifest_sha256"]
                m = json.loads(manifest.read_text())
                assert m["accepted"] == r["accepted_updates"]
                assert m["schema"] == r["native_schema"]
                for name, digest in m["artifacts"].items():
                    assert owned.sha(weight.parent / name) == digest
    rows = []
    for arm, opponent in p["tasks"]:
        for seed in p["match_seeds"]:
            tag = f"{seed}-{arm}-vs-{opponent}"
            old_invocation_path = args.parent_arena / (tag + ".invocation.json")
            old_result_path = args.parent_arena / (tag + ".json")
            if old_result_path.exists():
                original_owner = json.loads(
                    (args.parent_arena / (tag + ".owner-result.json")).read_text()
                )
                assert original_owner["status"] == "completed-games-awaiting-independent-audit"
                assert owned.sha(old_result_path) == original_owner["result_sha256"]
                assert original_owner["finished_epoch"] <= original_owner["original_deadline_epoch"]
                reused = {
                    **original_owner,
                    "result_path": str(old_result_path),
                    "original_owner_result_sha256": owned.sha(
                        args.parent_arena / (tag + ".owner-result.json")
                    ),
                    "reused_exact_closed_group": True,
                }
                rows.append(reused)
                owned.publish(args.output / (tag + ".reused-owner-result.json"), reused)
                continue
            if old_invocation_path.exists():
                original_invocation = json.loads(old_invocation_path.read_text())
                first = original_invocation["original_first_epoch"]
            else:
                first = time.time()
            deadline = min(end, first + p["per_tournament_seconds"])
            models = p["models"][str(seed)]
            cmd = [
                os.sys.executable,
                str(args.study / "tournament.py"),
                "--protocol",
                str(args.study / "protocol.json"),
                "--candidate",
                models[arm]["path"],
                "--opponent",
                opponent if opponent == "SF512" else models[opponent]["path"],
                "--seed",
                str(seed),
                "--book",
                p["book_path"],
                "--stockfish",
                str(Path("/workspace/work/harbichess/stockfish/stockfish-linux-x86-64-universal")),
                "--deadline-epoch",
                str(deadline),
                "--progress",
                str(args.output / (tag + ".progress.jsonl")),
                "--output",
                str(args.output / (tag + ".json")),
            ]
            if old_invocation_path.exists():
                assert deadline == original_invocation["original_deadline_epoch"]
                old_progress = args.parent_arena / (tag + ".progress.jsonl")
                assert old_progress.exists()
                cmd[1] = str(args.recovery_helper_dir / "resume_tournament.py")
                cmd += [
                    "--parent-progress",
                    str(old_progress),
                    "--parent-progress-sha256",
                    owned.sha(old_progress),
                    "--parent-invocation",
                    str(old_invocation_path),
                    "--parent-invocation-sha256",
                    owned.sha(old_invocation_path),
                    "--whole-deadline-epoch",
                    str(end),
                    "--old-parent-executor-sha256",
                    hashes["tournament.py"],
                ]
            row = {
                "seed": seed,
                "arm": arm,
                "opponent": opponent,
                "command": cmd,
                "result_path": str(args.output / (tag + ".json")),
                "recovery": old_invocation_path.exists(),
                "original_first_epoch": first,
                "original_deadline_epoch": deadline,
            }
            owned.publish(args.output / (tag + ".invocation.json"), row)
            try:
                assert time.time() < deadline
                assert (
                    owned.sha(args.recovery_helper_dir / "resume_tournament.py")
                    == contract["recovery_executor_sha256"]
                )
                assert (
                    owned.sha(args.recovery_helper_dir / "recovery_support.py")
                    == contract["recovery_support_sha256"]
                )
                assert all(
                    owned.sha(args.study / name) == digest for name, digest in hashes.items()
                )
                with (
                    (args.output / (tag + ".stdout.log")).open("xb") as out,
                    (args.output / (tag + ".stderr.log")).open("xb") as err,
                ):
                    owned.run_owned(
                        cmd,
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
                        deadline=deadline,
                    )
                result = json.loads((args.output / (tag + ".json")).read_text())
                assert len(result["games"]) == p["games_per_tournament"]
                row.update(
                    status="completed-games-awaiting-independent-audit",
                    result_sha256=owned.sha(args.output / (tag + ".json")),
                )
            except Exception as exc:
                row.update(status="failed-preserved", error=repr(exc))
            row["finished_epoch"] = time.time()
            owned.publish(args.output / (tag + ".owner-result.json"), row)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "command"}), flush=True)
    owned.publish(
        args.output / "cohort-result.json",
        {
            "status": "completed-games-not-strength"
            if all(r["status"].startswith("completed-") for r in rows)
            else "failed-preserved",
            "rows": rows,
            "contract": contract,
            "finished_epoch": time.time(),
        },
    )


if __name__ == "__main__":
    main()
