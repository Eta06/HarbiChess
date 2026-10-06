"""Small fail-closed CLI shell; running data preparation/fits is root-controlled."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from coarse80 import ARCH, PARAMS
from learner80 import OPTIMIZER

REG_SCHEMA = "own-search-coarse80-training-registration-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_registration(path):
    registration = json.loads(Path(path).read_text())
    expected = {
        "schema",
        "status",
        "seed",
        "cpu_core",
        "source",
        "q_labels",
        "prior",
        "search",
        "optimizer",
        "clocks",
        "output_dir",
        "closure",
    }
    if (
        set(registration) != expected
        or registration["schema"] != REG_SCHEMA
        or registration["status"] != "registered"
    ):
        raise ValueError("complete registered coarse80 protocol required")
    if type(registration["seed"]) is not int or type(registration["cpu_core"]) is not int:
        raise ValueError("integer seed and CPU core required")
    source = registration["source"]
    if set(source) != {"repository", "commit"} or len(source["commit"]) != 40:
        raise ValueError("pinned clean source commit required")
    try:
        int(source["commit"], 16)
    except ValueError as exc:
        raise ValueError("source commit must be a full hexadecimal hash") from exc
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source["repository"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source["repository"], text=True
    )
    if head != source["commit"] or dirty:
        raise ValueError("core source is not the registered clean commit")
    q = registration["q_labels"]
    if (
        set(q) != {"path", "sha256", "schema", "count"}
        or q["schema"] != "own-search-deeper-value-labels-v1"
        or q["count"] != 1024
        or sha(q["path"]) != q["sha256"]
    ):
        raise ValueError("frozen 1,024-row own-Q input changed")
    prior = registration["prior"]
    if set(prior) != {"helper_path", "helper_sha256", "model_path", "model_sha256"}:
        raise ValueError("frozen prior/helper pins required")
    for prefix in ("helper", "model"):
        if sha(prior[f"{prefix}_path"]) != prior[f"{prefix}_sha256"]:
            raise ValueError(f"frozen prior {prefix} changed")
    search = registration["search"]
    if set(search) != {
        "label_producer_path",
        "label_producer_sha256",
        "nodes",
        "q_depth",
        "max_depth",
    }:
        raise ValueError("search source/parameters must be pinned")
    if sha(search["label_producer_path"]) != search["label_producer_sha256"] or (
        search["nodes"],
        search["q_depth"],
        search["max_depth"],
    ) != (8192, 2, 8):
        raise ValueError("frozen own-search labeling contract differs")
    if registration["optimizer"] != {
        "architecture": ARCH,
        "parameters": PARAMS,
        "updates": 64,
        "batch_size": 256,
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "clip_norm": 5.0,
        "anchor_weight": 0.1,
        "ridge": 1e-4,
        "neighbor_smoothness": 1e-4,
        "loss": "mover_q_mse_only",
    }:
        raise ValueError("optimizer/target protocol differs")
    closure = registration["closure"]
    if not isinstance(closure, dict) or not closure:
        raise ValueError("complete local code closure required")
    for name, entry in closure.items():
        if set(entry) != {"path", "sha256"} or sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"code closure mismatch: {name}")
    clocks = registration["clocks"]
    if set(clocks) != {"proof", "fit"}:
        raise ValueError("separate original proof/fit clocks required")
    for name, duration in (("proof", 600.0), ("fit", 1800.0)):
        clock = clocks[name]
        if set(clock) != {"first", "deadline"}:
            raise ValueError(f"{name} clock fields differ")
        if not all(type(clock[k]) in (int, float) for k in ("first", "deadline")):
            raise ValueError(f"{name} clock not registered")
        if float(clock["deadline"]) != float(clock["first"]) + duration:
            raise ValueError(f"{name} original clock duration differs")
    if float(clocks["fit"]["first"]) < float(clocks["proof"]["deadline"]):
        raise ValueError("fit clock must begin after proof clock")
    output = Path(registration["output_dir"]).resolve()
    if not str(output).startswith("/dev/shm/harbichess-ownq-coarse80/"):
        raise ValueError("registered output must remain in the dedicated RAM area")
    return registration


def immutable_contract(registration):
    """Derive the complete native contract from a validated registration."""
    closure_bytes = json.dumps(
        registration["closure"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "schema": "ownq-coarse80-contract-v1",
        "seed": registration["seed"],
        "optimizer": OPTIMIZER,
        "labels_sha256": registration["q_labels"]["sha256"],
        "prior_sha256": registration["prior"]["model_sha256"],
        "search_producer_sha256": registration["search"]["label_producer_sha256"],
        "source_commit": registration["source"]["commit"],
        "closure_sha256": hashlib.sha256(closure_bytes).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Own-Q coarse80 residual protocol validator (no automatic fit)"
    )
    parser.add_argument("--registration", help="strict registered protocol JSON")
    parser.add_argument("--validate-registration", action="store_true")
    parser.add_argument(
        "--synthetic-native-proof",
        action="store_true",
        help="run only the embedded synthetic resume proof",
    )
    args = parser.parse_args()
    if args.synthetic_native_proof:
        from proof_synthetic import run

        print(json.dumps(run(), sort_keys=True))
        return
    if args.validate_registration:
        if not args.registration:
            parser.error("--registration is required")
        reg = validate_registration(args.registration)
        print(
            json.dumps(
                {"validated": True, "schema": reg["schema"], "seed": reg["seed"]}, sort_keys=True
            )
        )
        return
    parser.error(
        "choose --validate-registration or --synthetic-native-proof; "
        "production fit launch remains root-controlled"
    )


if __name__ == "__main__":
    main()
