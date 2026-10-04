"""Experiment-only native value guard; original oracle training/resume stays unchanged."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from harbichess.backends.torch_network import sha256
from harbichess.training import oracle_train
from harbichess.training.oracle_data import publish_json
from harbichess.training.torch_checkpoint import save_checkpoint


class NativeRetentionStop(RuntimeError):
    def __init__(self, measurement):
        self.measurement = measurement
        super().__init__("registered native value/Q retention boundary exceeded")


def guarded(
    directory: Path, dataset: Path, weights: Path, *, resume=None, stop_at=None, wall_seconds=1800
):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.perf_counter()
    profile = {
        "schema": 1,
        "guard_source_sha256": sha256(Path(__file__)),
        "oracle_source_sha256": sha256(Path(oracle_train.__file__)),
        "dataset_sha256": sha256(dataset / "dataset.json"),
        "initial_weights_sha256": sha256(weights),
        "value_ce_tolerance": 0.02,
        "qmae_tolerance": 0.02,
        "selection": "min policyCE among native-value-valid checkpoints",
        "patience_metric": "original oracle total policy+value CE",
        "seed": 20261017,
    }
    baseline = None
    if resume:
        resumed = json.loads((resume / "checkpoint.json").read_text())
        if resumed["run_state"].get("width_guard_retention_failed"):
            raise ValueError("failed retention checkpoint cannot continue this experiment")
        if json.loads((directory / "width-guard-profile.json").read_text()) != profile:
            raise ValueError("width guard profile changed; resume refused")
        initial = json.loads((directory / "checkpoints/step-000000/checkpoint.json").read_text())
        baseline = initial["run_state"]["evaluations"][0]
    original_evaluate = oracle_train.evaluate

    def checked(learner, panel):
        nonlocal baseline
        measurement = original_evaluate(learner, panel)
        if baseline is None:
            baseline = measurement
            publish_json(directory / "width-guard-profile.json", profile)
        invalid = (
            measurement["value_ce"] > baseline["value_ce"] + 0.02
            or measurement["value_mae"] > baseline["value_mae"] + 0.02
        )
        if invalid:
            previous = max(
                (directory / "checkpoints").glob("step-*"), key=lambda p: int(p.name.split("-")[1])
            )
            manifest = json.loads((previous / "checkpoint.json").read_text())
            state = manifest["run_state"]
            state["cursor"] = learner.step
            state["evaluations"].append(
                {"step": learner.step, **measurement, "width_guard_retention_failed": True}
            )
            state["width_guard_retention_failed"] = True
            paths = tuple(previous / name for name in manifest["replay"])
            checkpoint = directory / "checkpoints" / f"step-{learner.step:06d}"
            save_checkpoint(
                checkpoint,
                learner=learner,
                sampler=None,
                replay_paths=paths,
                run_state=state,
                run_config=manifest["run_config"],
                source_commit=source,
            )
            receipt = {
                "status": "failed",
                "step": learner.step,
                "native": measurement,
                "baseline": baseline,
                "last_complete_checkpoint": str(checkpoint),
                "source_commit": source,
                "profile": profile,
            }
            publish_json(directory / f"width-guard-failed-step-{learner.step:06d}.json", receipt)
            raise NativeRetentionStop(receipt)
        return measurement

    oracle_train.evaluate = checked
    try:
        core = oracle_train.run(
            directory,
            dataset,
            weights,
            resume=resume,
            stop_at=stop_at,
            max_steps=6000,
            interval=250,
            patience=1500,
            batch_size=64,
            wall_seconds=wall_seconds,
            seed=20261017,
            learning_rate=5e-5,
            max_train_rows=40000,
            max_validation_rows=20000,
        )
        report = {
            "status": core["status"],
            "core_result": core,
            "core_default_qualification_is_not_width_experiment_gate": True,
        }
    except NativeRetentionStop as stop:
        report = stop.measurement
    finally:
        oracle_train.evaluate = original_evaluate
    report = {
        **report,
        "source_commit": source,
        "wall_seconds": time.perf_counter() - started,
        "guard_profile": profile,
        "scope": (
            "Existing native loss/sampler/optimizer unchanged; explicit value guard "
            "and checkpoint selection are experiment-specific."
        ),
    }
    step = report.get("step", report.get("core_result", {}).get("step"))
    publish_json(directory / f"width-phase-result-step-{step:06d}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "dataset", "weights"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    result = guarded(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "guard_profile"}), flush=True)


if __name__ == "__main__":
    main()
