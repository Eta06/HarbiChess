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
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    p = json.loads((args.study / "protocol.json").read_text())
    owned.check_source(args.checkout, p["source_commit"])
    q = json.loads(args.qualification.read_text())
    assert q["status"] == "PASS-Coarse80-trained24-and-TRAIN48-parity-profile-not-strength"
    assert q["protocol_sha256"] == owned.sha(args.study / "protocol.json") and len(q["rows"]) == 24
    assert q["helper_sha256"] == owned.sha(args.study / "profile.py")
    parity = q["TRAIN48_parity"]
    assert parity["status"] == "PASS-TRAIN48-zeroHEX-and-trained-C80-Python-parity"
    assert {r["seed"] for r in parity["rows"]} == set(p["match_seeds"])
    assert len(parity["rows"]) == 2 and parity["tolerance"] == 1e-12
    for row in parity["rows"]:
        assert row["zero_hex_roots"] == row["trained_parity_roots"] == 24
        assert row["maximum_absolute_error"] <= 1e-12
        assert row["candidate_sha256"] == p["models"][str(row["seed"])]["learned"]["sha256"]
    assert len(q["latency_ratios"]) == 2
    assert all(
        r[k] <= 1.10
        for r in q["latency_ratios"]
        for k in ["learned_vs_original_fastprior", "learned_vs_original_E8"]
    )
    assert q["finished_epoch"] <= q["original_deadline_epoch"]
    assert args.first_epoch <= time.time()
    end = args.first_epoch + p["whole_seconds"]
    assert args.first_epoch == p["original_first_epoch"]
    assert end == p["original_deadline_epoch"]
    assert end <= p["operator_hard_deadline_epoch"]
    assert end > time.time()
    args.output.mkdir(exist_ok=False)
    hashes = {
        name: owned.sha(args.study / name)
        for name in ("protocol.json", "tournament.py", "search.py", "value.py")
    }
    contract = {
        "source_commit": p["source_commit"],
        "inputs": hashes,
        "helper_sha256": owned.sha(Path(__file__)),
        "qualification_sha256": owned.sha(args.qualification),
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": end,
        "GPU_used": False,
        "max_arena_workers": 1,
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
            if role == "learned":
                assert r["accepted_updates"] == 64
            elif role == "prior":
                model = json.loads(weight.read_text())
                assert model["schema"] == "classical-own-linear-value-v1"
                assert model["theta"] == [0.0] * 18
            elif role != "e8":
                raise ValueError("unknown model role")
    from native_admission import admit

    native_admissions = [admit(p, seed) for seed in p["match_seeds"]]
    owned.publish(args.output / "strict-native-admission.json", native_admissions)
    rows = []
    for arm, opponent in p["tasks"]:
        for seed in p["match_seeds"]:
            tag = f"{seed}-{arm}-vs-{opponent}"
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
                p["stockfish_path"],
                "--deadline-epoch",
                str(deadline),
                "--progress",
                str(args.output / (tag + ".progress.jsonl")),
                "--output",
                str(args.output / (tag + ".json")),
            ]
            row = {
                "seed": seed,
                "arm": arm,
                "opponent": opponent,
                "command": cmd,
                "original_first_epoch": first,
                "original_deadline_epoch": deadline,
            }
            owned.publish(args.output / (tag + ".invocation.json"), row)
            try:
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
