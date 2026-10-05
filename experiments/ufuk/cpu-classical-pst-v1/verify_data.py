"""Read-only proof that PST columns preserve the fixed classical18 datasets."""

import argparse
import json
import os
import time
from pathlib import Path

from frozen_reference.journal_v3 import sha as helper_sha
from prepare_pst import prepare


def publish(path, value):
    path = Path(path)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "classical-own-pst-offline-protocol-v1":
        raise ValueError("PST protocol schema differs")
    if args.output.exists():
        raise FileExistsError(args.output)
    first = time.time()
    receipts = {}
    for seed in protocol["seeds"]:
        key = str(seed)
        config_path = Path(protocol["config_paths"][key])
        journal_path = Path(protocol["journal_paths"][key])
        pair = protocol["inputs"][key]
        if helper_sha(config_path) != pair["config_sha256"]:
            raise ValueError(f"{seed}: config SHA differs")
        if helper_sha(journal_path) != pair["journal_sha256"]:
            raise ValueError(f"{seed}: fixed final journal SHA differs")
        config = json.loads(config_path.read_text())
        if config["seed"] != seed or config["max_actions"] != protocol["final_actions"]:
            raise ValueError(f"{seed}: actor config seed/fixed action count differs")
        if (
            config["excluded_training_position_keys"]
            != protocol["protected_position_keys"]
        ):
            raise ValueError(f"{seed}: protected positions differ")
        train, val, data_receipt, dataset_sha = prepare(
            journal_path, config, protocol["protected_position_keys"]
        )
        if (
            data_receipt["linear_dataset_sha256"]
            != pair["classical18_dataset_sha256"]
            or dataset_sha != pair["PST_dataset_sha256"]
        ):
            raise ValueError(f"{seed}: prepared dataset identity differs")
        if data_receipt["actions"] != protocol["final_actions"]:
            raise ValueError(f"{seed}: not the fixed final journal")
        if (
            sum(map(len, train.values())) != data_receipt["training_rows"]
            or sum(map(len, val.values())) != data_receipt["validation_rows"]
        ):
            raise ValueError(f"{seed}: frozen split row totals differ")
        receipts[key] = {
            "journal_path": str(journal_path),
            "journal_sha256": helper_sha(journal_path),
            "config_path": str(config_path),
            "config_sha256": helper_sha(config_path),
            "actions": data_receipt["actions"],
            "eligible_trajectories": data_receipt["eligible_games"],
            "eligible_rows": data_receipt["eligible_rows"],
            "training_rows": data_receipt["training_rows"],
            "validation_rows": data_receipt["validation_rows"],
            "updates": min(1024, 4 * data_receipt["training_rows"] // 256),
            "classical18_dataset_sha256": data_receipt["linear_dataset_sha256"],
            "PST_dataset_sha256": dataset_sha,
            "PST_columns": 224,
            "target_split_alignment": "PASS-exact-classical18-groups-features-and-targets",
        }
    output = {
        "schema": "classical-own-pst-dataset-preflight-v1",
        "status": "PASS-read-only-frozen-data-alignment-not-fit",
        "source_protocol_path": str(args.protocol),
        "source_protocol_sha256": helper_sha(args.protocol),
        "frozen_reference_helper_sha256": {
            name: helper_sha(Path(__file__).with_name("frozen_reference") / name)
            for name in ("learner.py", "journal_v3.py", "value.py")
        },
        "PST_helper_sha256": {
            name: helper_sha(Path(__file__).with_name(name))
            for name in (
                "pst_features.py",
                "pst_value.py",
                "pst_learner.py",
                "pst_native.py",
                "prepare_pst.py",
                "train_pst.py",
                "launch_pst.py",
            )
        },
        "seeds": receipts,
        "started_epoch": first,
        "finished_epoch": time.time(),
        "SGD": False,
        "searches": 0,
        "games": 0,
        "strength_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish(args.output, output)
    print(json.dumps({"status": output["status"], "seeds": len(receipts)}))


if __name__ == "__main__":
    main()
