"""Frozen-state guard for the preregistered native sparse-value experiment."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.training import oracle_train
from harbichess.training.oracle_data import publish_json


def frozen_hashes(network, prefixes):
    return {
        name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in network.named_parameters()
        if not name.startswith(prefixes)
    }


class FrozenStateStop(RuntimeError):
    pass


def guarded(
    directory: Path,
    dataset: Path,
    weights: Path,
    *,
    sparse=False,
    resume=None,
    stop_at=None,
    wall_seconds=1800,
):
    started = time.perf_counter()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    prefixes = (
        ("value_sparse_head.",) if sparse else ("value_", "invariant_value_", "global_value_")
    )
    initial = load_weights(weights)
    if bool(initial._value_sparse) != sparse:
        raise ValueError("sparse arm and initial specification disagree")
    expected = frozen_hashes(initial, prefixes)
    profile = {
        "schema": 1,
        "source_commit": source,
        "guard_source_sha256": sha256(Path(__file__)),
        "oracle_source_sha256": sha256(Path(oracle_train.__file__)),
        "dataset_sha256": sha256(dataset / "dataset.json"),
        "initial_weights_sha256": sha256(weights),
        "trainable_prefixes": list(prefixes),
        "frozen_hashes": expected,
        "seed": 20261022,
    }
    if resume:
        if json.loads((directory / "sparse-guard-profile.json").read_text()) != profile:
            raise ValueError("sparse guard profile changed; resume refused")
        if frozen_hashes(load_weights(resume / "model.safetensors"), prefixes) != expected:
            raise ValueError("frozen checkpoint changed; resume refused")
    del initial
    original_evaluate = oracle_train.evaluate

    def checked(learner, panel):
        actual = frozen_hashes(learner.network, prefixes)
        if actual != expected:
            receipt = {
                "status": "failed",
                "step": learner.step,
                "reason": "frozen state changed; no unsafe checkpoint published",
                "expected": expected,
                "actual": actual,
            }
            publish_json(directory / f"sparse-frozen-failed-step-{learner.step:06d}.json", receipt)
            raise FrozenStateStop(receipt["reason"])
        if not (directory / "sparse-guard-profile.json").exists():
            publish_json(directory / "sparse-guard-profile.json", profile)
        return original_evaluate(learner, panel)

    oracle_train.evaluate = checked
    try:
        core = oracle_train.run(
            directory,
            dataset,
            weights,
            resume=resume,
            stop_at=stop_at,
            max_steps=12000,
            interval=500,
            patience=2500,
            batch_size=64,
            wall_seconds=wall_seconds,
            seed=20261022,
            learning_rate=3e-4,
            max_train_rows=40000,
            max_validation_rows=20000,
            trainable_prefixes=prefixes,
        )
        report = {"status": core["status"], "core_result": core}
    except FrozenStateStop as stop:
        report = {"status": "failed", "reason": str(stop)}
    finally:
        oracle_train.evaluate = original_evaluate
    # Also inspect final publications when a wall stop falls between evaluations.
    checkpoints = sorted((directory / "checkpoints").glob("step-*"))
    for checkpoint in checkpoints:
        if frozen_hashes(load_weights(checkpoint / "model.safetensors"), prefixes) != expected:
            report = {"status": "failed", "reason": "published frozen-state mismatch"}
            break
    step = int(checkpoints[-1].name.split("-")[1]) if checkpoints else 0
    report = {
        **report,
        "step": step,
        "source_commit": source,
        "wall_seconds": time.perf_counter() - started,
        "published_checkpoints_audited": len(checkpoints),
        "core_default_qualification_is_not_sparse_experiment_gate": True,
    }
    publish_json(directory / f"sparse-phase-result-step-{step:06d}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "dataset", "weights"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    print(json.dumps(guarded(**vars(parser.parse_args()))), flush=True)


if __name__ == "__main__":
    main()
