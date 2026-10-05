"""Prospective fixed600 CPU full training restart proof, no strength claim."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b, strict=True))
        )
    if hasattr(a, "dtype") and hasattr(a, "tobytes"):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "trainer", "protocol", "weights", "journal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if args.deadline_epoch != args.first_epoch + 600 or args.first_epoch > time.time():
        raise ValueError("fixed original600 clock required")
    args.output.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "cpu_owned", args.checkout / "experiments/ufuk/cpu-fix-v1/qualify_cli.py"
    )
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    shared = [
        sys.executable,
        str(args.trainer),
        "--weights",
        str(args.weights),
        "--journal",
        str(args.journal),
        "--protocol",
        str(args.protocol),
        "--source-commit",
        args.source_commit,
        "--seed",
        str(args.seed),
        "--steps",
        "8",
        "--deadline-epoch",
        str(args.deadline_epoch),
    ]
    result = {
        "schema": "cpu-own-outcome-positional-restart-proof-v2",
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": args.deadline_epoch,
        "source_commit": args.source_commit,
        "scope": (
            "training-only checkpoint/optimizer/RNG/replay proof; no online actor or strength claim"
        ),
    }
    try:
        for name, directory, extra in [
            ("whole", "whole", []),
            ("pause", "split", ["--stop-at", "4"]),
            ("resume", "split", ["--resume", str(args.output / "split/checkpoints/step-00000004")]),
        ]:
            with (
                (args.output / (name + ".stdout.log")).open("xb") as out,
                (args.output / (name + ".stderr.log")).open("xb") as err,
            ):
                owned.run_owned(
                    [*shared, "--output", str(args.output / directory), *extra],
                    cwd=args.checkout,
                    env={**os.environ, "PYTHONPATH": str(args.checkout / "src")},
                    stdout=out,
                    stderr=err,
                    deadline=args.deadline_epoch,
                )
        snapshots = []
        for name in ("whole", "split"):
            root = args.output / name / "checkpoints/step-00000008"
            manifest = json.loads((root / "checkpoint.json").read_text())
            assert manifest["schema"] == "cpu-own-outcome-positional-training-native-v2"
            for filename, digest in manifest["artifacts"].items():
                assert owned.sha(root / filename) == digest
            snapshots.append(
                torch.load(root / "training.pt", map_location="cpu", weights_only=True)
            )
        assert equal(*snapshots), "model/optimizer/global+sampler RNG/data contract differ"
        assert snapshots[0]["accepted"] == snapshots[0]["attempted"] == 8
        assert snapshots[0]["contract"]["device"] == "cpu"
        # Strict fresh-process loading of each full native: no update or old-file writes.
        for name in ("whole", "split"):
            for step in (0, 4, 8):
                tag = f"strict-{name}-{step}"
                with (
                    (args.output / (tag + ".stdout.log")).open("xb") as out,
                    (args.output / (tag + ".stderr.log")).open("xb") as err,
                ):
                    owned.run_owned(
                        [
                            *shared,
                            "--output",
                            str(args.output / name),
                            "--resume",
                            str(args.output / name / f"checkpoints/step-{step:08d}"),
                            "--audit-only",
                        ],
                        cwd=args.checkout,
                        env={**os.environ, "PYTHONPATH": str(args.checkout / "src")},
                        stdout=out,
                        stderr=err,
                        deadline=args.deadline_epoch,
                    )
        result.update(
            status="pass",
            all6full_native_freshprocess_strictload=True,
            all_training_payload_fields_bit_identical=True,
            accepted_updates=8,
            full_optimizer_rng_replay_contract_resume=True,
            native_inputs_sha256=snapshots[0]["contract"]["journal_sha256"],
        )
    except Exception as e:
        result.update(status="failed-preserved", error=repr(e))
    result["finished_epoch"] = time.time()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
