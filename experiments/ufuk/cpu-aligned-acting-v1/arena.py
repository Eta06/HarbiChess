"""One owned CPU arena worker; admission needs each fresh native/data audit."""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "protocol", "weights", "stockfish", "fits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    protocol = json.loads(args.protocol.read_text())
    owned.check_source(args.checkout, protocol["source_commit"])
    if args.first_epoch > time.time():
        raise ValueError("observed original clock required")
    deadline = args.first_epoch + protocol["whole_seconds"]
    book = Path(protocol["book_path"])
    assert owned.sha(book) == protocol["book_sha256"]
    assert owned.sha(args.weights) == protocol["initial_e8_sha256"]
    assert owned.sha(args.stockfish) == protocol["stockfish_sha256"]
    args.output.mkdir(exist_ok=False)
    contract = {
        "protocol_sha256": owned.sha(args.protocol),
        "helper_sha256": owned.sha(Path(__file__)),
        "original_first_epoch": args.first_epoch,
        "original_whole_deadline_epoch": deadline,
        "max_arena_workers": 1,
        "GPU_used": False,
        "scope": "known development suite, not independent formal confirmation",
    }
    owned.publish(args.output / "owner-contract.json", contract)
    tasks = [(seed, "e8", "SF512") for seed in protocol["seeds"]]
    tasks += [
        (seed, arm, opponent)
        for arm in protocol["arms"]
        for seed in protocol["seeds"]
        for opponent in ("e8", "SF512")
    ]
    rows = []
    for seed, arm, opponent in tasks:
        tag = f"{seed}-{arm}-vs-{opponent}"
        if arm != "e8":
            audit = args.fits / f"{seed}-{arm}.fresh-audit.json"
            failed = args.fits / f"{seed}-{arm}.audit-owner-result.json"
            while not audit.exists() and time.time() < deadline:
                if (
                    failed.exists()
                    and json.loads(failed.read_text())["status"] == "failed-preserved"
                ):
                    break
                time.sleep(2)
            if not audit.exists():
                row = {"tag": tag, "status": "not-admitted-no-data-audit-PASS"}
                owned.publish(args.output / (tag + ".owner-result.json"), row)
                rows.append(row)
                continue
            receipt = json.loads(audit.read_text())
            assert receipt["status"].startswith("pass-data-")
            assert receipt["source_commit"] == protocol["source_commit"]
            candidate = args.fits / f"{seed}-{arm}/checkpoints/epoch-00000003/model.safetensors"
            assert owned.sha(candidate) == receipt["final_artifacts"]["model.safetensors"]
        else:
            candidate = args.weights
        first = time.time()
        end = min(first + protocol["per_tournament_seconds"], deadline)
        if end <= first:
            raise TimeoutError("original arena campaign clock exhausted")
        command = [
            os.sys.executable,
            "-m",
            "harbichess.evaluation.portable_arena",
            str(candidate),
            str(args.weights) if opponent == "e8" else "stockfish",
            "--stockfish",
            str(args.stockfish),
            "--simulations",
            str(protocol["simulations"]),
            "--nodes",
            str(protocol["stockfish_nodes"]),
            "--max-plies",
            str(protocol["max_plies"]),
            "--wall-seconds",
            str(end - first),
            "--opening-pairs",
            str(protocol["opening_pairs"]),
            "--openings",
            str(book),
            "--seed",
            str(seed),
            "--threads",
            "1",
            "--candidate-root-actions",
            str(protocol["root_actions"]),
            "--opponent-root-actions",
            str(protocol["root_actions"]),
            "--record-engine-nodes",
            "--progress",
            str(args.output / (tag + ".progress.jsonl")),
            "--output",
            str(args.output / (tag + ".json")),
        ]
        row = {
            "tag": tag,
            "seed": seed,
            "arm": arm,
            "opponent": opponent,
            "original_first_epoch": first,
            "original_deadline_epoch": end,
            "candidate_sha256": owned.sha(candidate),
            "command": command,
        }
        owned.publish(args.output / (tag + ".invocation.json"), row)
        try:
            assert owned.sha(args.protocol) == contract["protocol_sha256"]
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
            result = json.loads((args.output / (tag + ".json")).read_text())
            assert len(result["games"]) == protocol["games_per_tournament"]
            assert result["source_commit"] == protocol["source_commit"]
            row.update(
                status="completed-games-awaiting-fullhistory-replay",
                result_sha256=owned.sha(args.output / (tag + ".json")),
            )
        except BaseException as exc:
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
