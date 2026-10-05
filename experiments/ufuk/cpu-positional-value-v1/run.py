"""Sequential CPU critic fits and strict native loads, after actual restart admission."""

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "study", "weights", "qualification", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    p = json.loads((args.study / "protocol.json").read_text())
    owned.check_source(args.checkout, p["source_commit"])
    q = json.loads((args.qualification / "result.json").read_text())
    assert q["status"] == "pass" and q["all6full_native_freshprocess_strictload"]
    assert q["source_commit"] == p["source_commit"]
    assert args.first_epoch == q["original_first_epoch"]
    deadline = args.first_epoch + p["whole_fit_seconds"]
    if time.time() >= deadline:
        raise TimeoutError("original study fit budget exhausted")
    args.output.mkdir(exist_ok=False)
    hashes = {
        name: owned.sha(args.study / name) for name in ("protocol.json", "train.py", "features.py")
    }
    contract = {
        "source_commit": p["source_commit"],
        "inputs": hashes,
        "helper_sha256": owned.sha(Path(__file__)),
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": deadline,
        "max_cpu_fit_workers": 1,
        "GPU_used": False,
    }
    owned.publish(args.output / "owner-contract.json", contract)
    free = shutil.disk_usage(args.output).free
    required = p["disk_min_free_bytes"] + 1.5 * 32 * 1024**2
    owned.publish(
        args.output / "resource-admission.json",
        {
            "free_bytes": free,
            "required_bytes": required,
            "reserved_bytes": 32 * 1024**2,
            "safety_factor": 1.5,
            "admitted": free >= required,
            "scope": (
                "two small native fits and games; replay features held only in RAM; "
                "no oldfile deletion"
            ),
        },
    )
    if free < required:
        raise RuntimeError("new critic fit disk admission failed")
    env = {
        **os.environ,
        "PYTHONPATH": str(args.checkout / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    rows = []
    for seed in p["seeds"]:
        first = time.time()
        end = min(deadline, first + p["per_fit_seconds"])
        tag = f"{seed}-positional"
        root = args.output / tag
        command = [
            os.sys.executable,
            str(args.study / "train.py"),
            "--weights",
            str(args.weights),
            "--protocol",
            str(args.study / "protocol.json"),
            "--source-commit",
            p["source_commit"],
            "--seed",
            str(seed),
            "--steps",
            str(p["steps"]),
            "--deadline-epoch",
            str(end),
            "--output",
            str(root),
        ]
        source_seed = p["pairing"][str(seed)]
        for record in p["journals"]:
            if record["source_seed"] == source_seed:
                command += ["--journal", record["path"]]
        row = {
            "seed": seed,
            "tag": tag,
            "command": command,
            "original_first_epoch": first,
            "original_deadline_epoch": end,
        }
        owned.publish(args.output / (tag + ".invocation.json"), row)
        try:
            assert all(owned.sha(args.study / name) == h for name, h in hashes.items())
            with (
                (args.output / (tag + ".stdout.log")).open("xb") as out,
                (args.output / (tag + ".stderr.log")).open("xb") as err,
            ):
                owned.run_owned(
                    command, cwd=args.checkout, env=env, stdout=out, stderr=err, deadline=end
                )
            results = list(root.glob(f"result-step-{p['steps']:08d}-invocation-*.json"))
            assert len(results) == 1
            fit = json.loads(results[0].read_text())
            assert (
                fit["status"] == "completed-fit-not-strength"
                and fit["accepted_updates"] == p["steps"]
            )
            # Read-only strict fresh loads retain the exact original training deadline.
            for step in (0, p["steps"]):
                name = tag + f"-native-{step}"
                with (
                    (args.output / (name + ".stdout.log")).open("xb") as out,
                    (args.output / (name + ".stderr.log")).open("xb") as err,
                ):
                    owned.run_owned(
                        [
                            *command,
                            "--resume",
                            str(root / f"checkpoints/step-{step:08d}"),
                            "--audit-only",
                        ],
                        cwd=args.checkout,
                        env=env,
                        stdout=out,
                        stderr=err,
                        deadline=end,
                    )
            manifests = {
                str(step): json.loads(
                    (root / f"checkpoints/step-{step:08d}/checkpoint.json").read_text()
                )
                for step in (0, p["steps"])
            }
            row.update(
                status="PASS-fit-native-not-strength",
                result_sha256=owned.sha(results[0]),
                result=fit,
                initial_manifest=manifests["0"],
                final_manifest=manifests[str(p["steps"])],
                initial_and_final_fullnative_freshprocess_loads=True,
            )
        except BaseException as exc:
            row.update(status="failed-preserved", error=repr(exc))
        row["finished_epoch"] = time.time()
        owned.publish(args.output / (tag + ".owner-result.json"), row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("tag", "status", "finished_epoch")}), flush=True)
    owned.publish(
        args.output / "cohort-result.json",
        {
            "status": "PASS-fits-native-not-strength"
            if all(r["status"].startswith("PASS-") for r in rows)
            else "failed-preserved",
            "rows": rows,
            "contract": contract,
            "finished_epoch": time.time(),
            "strength_success_claimed": False,
            "GPU_used": False,
        },
    )


if __name__ == "__main__":
    main()
