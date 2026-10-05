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
    cohort = json.loads((args.fits / "cohort-result.json").read_text())
    assert cohort["status"] == "PASS-fits-native-not-strength"
    assert cohort["contract"]["source_commit"] == protocol["source_commit"]
    tasks = [
        (seed, arm, opponent) for arm, opponent in protocol["tasks"] for seed in protocol["seeds"]
    ]

    def model_path(seed, role):
        if role == "e8":
            return args.weights
        step = 0 if role == "rebased" else 4096
        return (
            args.fits / f"{seed}-positional/checkpoints" / f"step-{step:08d}" / "model.safetensors"
        )

    for seed in protocol["seeds"]:
        row = next(r for r in cohort["rows"] if r["seed"] == seed)
        assert row["status"] == "PASS-fit-native-not-strength"
        assert row["initial_and_final_fullnative_freshprocess_loads"]
        for role, key in (("rebased", "initial_manifest"), ("positional", "final_manifest")):
            assert owned.sha(model_path(seed, role)) == row[key]["artifacts"]["model.safetensors"]
            assert row[key]["accepted"] == (0 if role == "rebased" else 4096)
    rows = []
    for seed, arm, opponent in tasks:
        tag = f"{seed}-{arm}-vs-{opponent}"
        candidate = model_path(seed, arm)
        first = time.time()
        end = min(first + protocol["per_tournament_seconds"], deadline)
        if end <= first:
            raise TimeoutError("original arena campaign clock exhausted")
        command = [
            os.sys.executable,
            "-m",
            "harbichess.evaluation.portable_arena",
            str(candidate),
            str(model_path(seed, opponent)) if opponent != "SF512" else "stockfish",
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
