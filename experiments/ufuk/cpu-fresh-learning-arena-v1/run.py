"""Root-invoked owner for the fixed fresh own-learning CPU arena."""

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

from contracts import (
    load_fit_manifest,
    model_record,
    sha,
    verify_fit_inputs,
    verify_fixed_arena_protocol,
    verify_profile_receipt,
)


def load_owner(checkout):
    path = checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    spec = importlib.util.spec_from_file_location("fresh_arena_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "protocol", "qualification", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--first-epoch", type=float, required=True)
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_fixed_arena_protocol(p)
    if p.get("status") != "registered":
        raise ValueError("arena protocol must be frozen and registered")
    owner = load_owner(args.checkout)
    owner.check_source(args.checkout, p["source_commit"])
    q = json.loads(args.qualification.read_text())
    fit_path, fit = load_fit_manifest(p)
    verify_fit_inputs(p, fit)
    verify_profile_receipt(
        p,
        q,
        sha(args.protocol),
        sha(fit_path),
        sha(Path(__file__).with_name("profile.py")),
        fit,
    )
    q_first = q["original_first_epoch"]
    q_deadline = q["original_deadline_epoch"]
    end = args.first_epoch + p["whole_seconds"]
    if (
        args.first_epoch != p["original_first_epoch"]
        or args.first_epoch > time.time()
        or end != p["original_deadline_epoch"]
        or end > p["operator_hard_deadline_epoch"]
        or time.time() >= end
        or q_first != p["profile_first_epoch"]
        or q_deadline != p["profile_deadline_epoch"]
    ):
        raise ValueError("fixed original arena clock is invalid or expired")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    helpers = (
        "protocol.json",
        "fit-provenance.json",
        "tournament.py",
        "search.py",
        "value.py",
        "contracts.py",
    )
    inputs = {
        name: sha(
            args.protocol
            if name == "protocol.json"
            else fit_path
            if name == "fit-provenance.json"
            else Path(__file__).with_name(name)
        )
        for name in helpers
    }
    contract = {
        "schema": "fresh-own-learning-arena-owner-v1",
        "source_commit": p["source_commit"],
        "inputs": inputs,
        "run_helper_sha256": sha(Path(__file__)),
        "profile_sha256": sha(args.qualification),
        "fit_manifest_sha256": sha(fit_path),
        "first_epoch": args.first_epoch,
        "deadline_epoch": end,
        "GPU_used": False,
        "workers": 1,
    }
    owner.publish(args.output / "owner-contract.json", contract)
    free = shutil.disk_usage(args.output).free
    required = p["disk_min_free_bytes"] + p["reserved_output_bytes"]
    owner.publish(
        args.output / "resource-admission.json",
        {"free_bytes": free, "required_bytes": required, "admitted": free >= required},
    )
    if free < required:
        raise RuntimeError("registered no-deletion disk admission failed")
    tasks = [(s, arm, opp) for s in p["match_seeds"] for arm, opp in p["tasks"]]
    if len(tasks) != 14:
        raise ValueError("fixed seven tournaments per seed required")
    rows = []
    for seed, arm, opponent in tasks:
        tag = f"{seed}-{arm}-vs-{opponent}"
        started = time.time()
        deadline = min(end, started + p["per_tournament_seconds"])
        models = {r: model_record(p, fit, seed, r) for r in ("e8", "mc", "sc", "full")}
        cmd = [
            os.sys.executable,
            str(Path(__file__).with_name("tournament.py")),
            "--protocol",
            str(args.protocol),
            "--candidate",
            models[arm]["path"],
            "--opponent",
            opponent if opponent == "SF512" else models[opponent]["path"],
            "--seed",
            str(seed),
            "--book",
            p["book_path"],
            "--stockfish",
            p["stockfish_path"],
            "--deadline-epoch",
            str(deadline),
            "--progress",
            str(args.output / f"{tag}.progress.jsonl"),
            "--output",
            str(args.output / f"{tag}.json"),
        ]
        row = {
            "seed": seed,
            "arm": arm,
            "opponent": opponent,
            "command": cmd,
            "original_first_epoch": started,
            "original_deadline_epoch": deadline,
        }
        owner.publish(args.output / f"{tag}.invocation.json", row)
        try:
            if time.time() >= end:
                raise TimeoutError("original whole-arena deadline expired")
            if any(
                sha(Path(__file__).with_name(n)) != digest
                for n, digest in inputs.items()
                if n not in {"protocol.json", "fit-provenance.json"}
            ):
                raise ValueError("arena helper changed during cohort")
            with (
                (args.output / f"{tag}.stdout.log").open("xb") as out,
                (args.output / f"{tag}.stderr.log").open("xb") as err,
            ):
                owner.run_owned(
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
            result_path = args.output / f"{tag}.json"
            result = json.loads(result_path.read_text())
            if len(result.get("games", [])) != p["games_per_tournament"]:
                raise ValueError("tournament did not produce 16 games")
            row.update(
                status="completed-games-awaiting-independent-audit",
                result_sha256=sha(result_path),
            )
        except Exception as exc:
            row.update(status="failed-preserved", error=repr(exc))
        row["finished_epoch"] = time.time()
        owner.publish(args.output / f"{tag}.owner-result.json", row)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "command"}), flush=True)
    on_time = time.time() < end
    if not on_time:
        rows.append(
            {
                "status": "failed-preserved",
                "error": "original whole-arena deadline expired before final cohort receipt",
            }
        )
    owner.publish(
        args.output / "cohort-result.json",
        {
            "status": "completed-games-not-strength"
            if len(rows) == 14
            and all(r["status"].startswith("completed-") for r in rows)
            and on_time
            else "failed-preserved",
            "rows": rows,
            "contract": contract,
            "finished_epoch": time.time(),
        },
    )


if __name__ == "__main__":
    main()
