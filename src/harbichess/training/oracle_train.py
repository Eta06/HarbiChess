"""Bounded soft-target bootstrap with whole-family validation and full resume."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.training.config import LearnerConfig
from harbichess.training.oracle_data import ORACLE_SCHEMA, load_panels, publish_json
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch


def evaluate(learner: TorchLearner, panel: TorchTrainingBatch) -> dict:
    totals = {k: 0.0 for k in ("policy_ce", "value_ce", "value_mae", "top16_coverage")}
    with torch.no_grad():
        for start in range(0, panel.size, 64):
            batch = panel.select(tuple(range(start, min(panel.size, start + 64))))
            policy, wdl = learner.network(batch.inputs)
            masked = policy.masked_fill(~batch.legal_masks, -1e9)
            totals["policy_ce"] += float(-(batch.policies * F.log_softmax(masked, 1)).sum())
            totals["value_ce"] += float(F.cross_entropy(wdl, batch.wdl, reduction="sum"))
            predicted = wdl.softmax(1)
            totals["value_mae"] += float(
                ((predicted[:, 0] - predicted[:, 2]) - (batch.wdl[:, 0] - batch.wdl[:, 2]))
                .abs()
                .sum()
            )
            best = batch.policies.argmax(1).unsqueeze(1)
            totals["top16_coverage"] += float((masked.topk(16, 1).indices == best).any(1).sum())
    result = {k: value / panel.size for k, value in totals.items()}
    return {**result, "total_ce": result["policy_ce"] + result["value_ce"], "rows": panel.size}


def run(
    directory: Path,
    dataset: Path,
    weights: Path,
    *,
    resume: Path | None = None,
    stop_at: int | None = None,
    max_steps: int = 6000,
    interval: int = 200,
    patience: int = 1000,
    batch_size: int = 64,
    wall_seconds: float = 1800,
    seed: int = 20261003,
    trainable_prefixes: tuple[str, ...] = (),
    qualification_reference: Path | None = None,
    learning_rate: float = 2e-4,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
) -> dict:
    if min(max_steps, interval, patience, batch_size, wall_seconds) <= 0:
        raise ValueError("training budgets must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    training, validation, panel_info = load_panels(
        dataset, max_train_rows=max_train_rows, max_validation_rows=max_validation_rows, seed=seed
    )
    learner_config = LearnerConfig(learning_rate=learning_rate)
    config = {
        "task": "engine-reference-bootstrap",
        "oracle_schema": ORACLE_SCHEMA,
        "target_semantics": "native STM soft reference WDL, not observed outcome replay",
        "dataset_sha256": sha256(dataset / "dataset.json"),
        "initial_weights_sha256": sha256(weights),
        "seed": seed,
        "max_steps": max_steps,
        "interval": interval,
        "patience": patience,
        "batch_size": batch_size,
        "learner": asdict(learner_config),
        "code_sha256": {
            p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("oracle_data.py"))
        },
    }
    if trainable_prefixes:
        config["trainable_prefixes"] = list(trainable_prefixes)
    if qualification_reference:
        config["qualification_reference_sha256"] = sha256(qualification_reference)
    if max_train_rows is not None or max_validation_rows is not None:
        config["panel_caps"] = {"train": max_train_rows, "validation": max_validation_rows}
    if resume:
        learner, manifest, _ = load_checkpoint(resume, expected_run_config=config)
        state = manifest["run_state"]
        if manifest["step"] != state["cursor"]:
            raise ValueError("checkpoint learning cursor mismatch")
        if resume.parent.parent.resolve() != directory.resolve():
            raise ValueError("resume in the original complete run directory")
    else:
        if directory.exists():
            raise FileExistsError("new learning runs require a fresh directory")
        directory.mkdir(parents=True)
        torch.manual_seed(seed)
        network = load_weights(weights)
        if trainable_prefixes:
            for name, parameter in network.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
            if not any(p.requires_grad for p in network.parameters()):
                raise ValueError("trainable prefixes match no parameters")
        learner = TorchLearner(network, config=learner_config)
        baseline = evaluate(learner, validation)
        state = {
            "cursor": 0,
            "best_step": 0,
            "best_score": baseline["total_ce"],
            "evaluations": [{"step": 0, **baseline}],
        }
        if qualification_reference:
            state["qualification_reference"] = evaluate(
                TorchLearner(load_weights(qualification_reference)), validation
            )
        publish_json(
            directory / "metadata.json",
            {"source_commit": source, "config": config, "panels": panel_info},
        )
    inputs = (
        *sorted(dataset.glob("*.json.gz")),
        dataset / "dataset.json",
        dataset / "metadata.json",
    )

    def checkpoint() -> None:
        state["cursor"] = learner.step
        path = directory / "checkpoints" / f"step-{learner.step:06d}"
        if not path.exists():
            save_checkpoint(
                path,
                learner=learner,
                sampler=None,
                replay_paths=inputs,
                run_state=state,
                run_config=config,
                source_commit=source,
            )

    checkpoint()
    print(
        json.dumps(
            {
                "event": "ready",
                "step": learner.step,
                "panels": panel_info,
                "validation": state["evaluations"][-1],
            }
        ),
        flush=True,
    )
    stop = min(max_steps, stop_at if stop_at is not None else max_steps)
    reason = "maximum updates"
    with (directory / "progress.jsonl").open("a") as log:
        while learner.step < stop:
            if time.perf_counter() - started >= wall_seconds:
                reason = "wall budget"
                break
            if learner.step - state["best_step"] >= patience:
                reason = "validation early stop"
                break
            indices = tuple(torch.randint(training.size, (batch_size,)).tolist())
            metrics = learner.train_step(training.select(indices))
            if learner.step % interval == 0:
                measured = evaluate(learner, validation)
                state["evaluations"].append({"step": learner.step, **measured})
                if measured["total_ce"] < state["best_score"] - 1e-4:
                    state.update(best_step=learner.step, best_score=measured["total_ce"])
                checkpoint()
                record = {
                    "step": learner.step,
                    "training_last_batch": asdict(metrics),
                    "validation": measured,
                    "best_step": state["best_step"],
                    "wall_seconds": time.perf_counter() - started,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    if stop_at is not None and learner.step == stop_at and stop_at < max_steps:
        reason = "registered process boundary"
    checkpoint()
    baseline = state.get("qualification_reference", state["evaluations"][0])
    best = next(row for row in state["evaluations"] if row["step"] == state["best_step"])
    qualified = (
        baseline["policy_ce"] - best["policy_ce"] >= 0.10
        and baseline["value_ce"] - best["value_ce"] >= 0.10
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema": 1,
        "source_commit": source,
        "status": "completed"
        if reason in ("maximum updates", "validation early stop")
        else "paused",
        "reason": reason,
        "step": learner.step,
        "best_step": state["best_step"],
        "best_weights_sha256": sha256(
            directory / "checkpoints" / f"step-{state['best_step']:06d}" / "model.safetensors"
        ),
        "initial_validation": baseline,
        "best_validation": best,
        "qualification_for_arena": qualified,
        "promotion_ready": False,
        "wall_seconds_this_process": time.perf_counter() - started,
        "self_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "self_maxrss_kib": usage.ru_maxrss,
        "resume_from": str(resume) if resume else None,
    }
    publish_json(directory / f"result-step-{learner.step:06d}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--trainable-prefix", action="append", default=[])
    parser.add_argument("--qualification-reference", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.directory,
                args.dataset,
                args.weights,
                resume=args.resume,
                stop_at=args.stop_at,
                wall_seconds=args.wall_seconds,
                trainable_prefixes=tuple(args.trainable_prefix),
                qualification_reference=args.qualification_reference,
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                interval=args.interval,
                patience=args.patience,
                seed=args.seed,
                max_train_rows=args.max_train_rows,
                max_validation_rows=args.max_validation_rows,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
