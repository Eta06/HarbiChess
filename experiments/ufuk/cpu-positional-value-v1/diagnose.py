"""Registered post-hoc calibration curve, not training or endpoint selection."""

import argparse
import json
import time
from pathlib import Path

import torch
from features import prepare, sha
from train import metrics

from harbichess.backends.torch_network import load_weights
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_run import clean_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "fits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    clean_source(p["source_commit"])
    torch.set_num_threads(1)
    assert args.first_epoch <= time.time() < args.first_epoch + p["whole_seconds"]
    budget = CgroupMemoryBudget(p["memory_max_bytes"])
    rows = []
    for seed in p["seeds"]:
        deadline = min(args.first_epoch + p["whole_seconds"], time.time() + p["per_seed_seconds"])

        def guard(deadline=deadline):
            if time.time() >= deadline:
                raise TimeoutError("original diagnostic clock exhausted")
            budget.check()

        paths = [
            Path(j["path"]) for j in p["journals"] if j["source_seed"] == p["pairing"][str(seed)]
        ]
        expected = [
            j["sha256"] for j in p["journals"] if j["source_seed"] == p["pairing"][str(seed)]
        ]
        assert [sha(j) for j in paths] == expected
        x, y, train, val, receipts, digest = prepare(paths, guard)
        for step in p["steps"]:
            guard()
            root = args.fits / f"{seed}-positional/checkpoints/step-{step:08d}"
            manifest = json.loads((root / "checkpoint.json").read_text())
            assert manifest["accepted"] == step and manifest["artifacts"][
                "model.safetensors"
            ] == sha(root / "model.safetensors")
            model = load_weights(root / "model.safetensors").eval()
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "model_sha256": sha(root / "model.safetensors"),
                    "dataset_sha256": digest,
                    "journal_receipts": receipts,
                    "train": metrics(model, x, y, train),
                    "validation": metrics(model, x, y, val),
                    "finished_epoch": time.time(),
                }
            )
        del x, y, train, val, model
    result = {
        "status": "completed-diagnostic-not-strength",
        "rows": rows,
        "protocol_sha256": sha(args.protocol),
        "helper_sha256": sha(Path(__file__)),
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": args.first_epoch + p["whole_seconds"],
        "finished_epoch": time.time(),
        "GPU_used": False,
        "strength_success_claimed": False,
        "endpoint_selected": False,
    }
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({"status": result["status"], "rows": len(rows)}))


if __name__ == "__main__":
    main()
