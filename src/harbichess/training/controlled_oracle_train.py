"""Joint soft-target training with common, independently reported validation groups.

Both data arms must use the same immutable validation source, whose training keys
cover either arm. This keeps overlap removal identical before group selection.
Equal group mean policy+value CE selects checkpoints; chess strength is external.
"""

import argparse
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from harbichess.backends import torch_network
from harbichess.chess import packed_encoding
from harbichess.training import (
    oracle_data,
    oracle_train,
    prepared_oracle,
    prepared_oracle_torch,
    torch_checkpoint,
    torch_learner,
)
from harbichess.training.config import LearnerConfig


def common_panels(training_cache, validation_cache, groups, *, train_cap, validation_cap, seed):
    training, _, training_info = training_cache.panels(max_train_rows=train_cap, seed=seed)
    if not set(training_cache.position_key[training_cache.split == 0]) <= set(
        validation_cache.position_key[validation_cache.split == 0]
    ):
        raise ValueError("common validation source must cover all arm training keys")
    _, validation, info = validation_cache.panels(max_train_rows=1, seed=seed)
    if not groups or len({name for name, _, _ in groups}) != len(groups):
        raise ValueError("nonempty unique validation groups required")
    panels, receipts, assigned = {}, {}, set()
    for name, lower, upper in groups:
        if not name.isidentifier() or not 0 <= lower < upper <= 2**32:
            raise ValueError("invalid named validation family range")
        rows = [int(i) for i in validation.rows if lower <= validation_cache.family[i] < upper]
        if not rows or assigned.intersection(rows):
            raise ValueError("validation groups must be nonempty and disjoint")
        assigned.update(rows)
        available = len(rows)
        if validation_cap is not None:
            if type(validation_cap) is not int or validation_cap <= 0:
                raise ValueError("positive per-group validation cap required")
            if len(rows) > validation_cap:
                rng = random.Random(f"{seed}:validation:{name}")
                rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), validation_cap))]
        panels[name] = prepared_oracle_torch.TorchPreparedPanel(
            prepared_oracle.PreparedPanel(validation_cache, rows)
        )
        receipts[name] = {
            "available_rows": available,
            "rows": len(rows),
            "families": sorted({int(validation_cache.family[i]) for i in rows}),
            "row_indices_sha256": hashlib.sha256(np.array(rows, dtype="<u8").tobytes()).hexdigest(),
        }
    if assigned != set(validation.rows.tolist()):
        raise ValueError("validation groups must cover every retained validation row")
    return (
        prepared_oracle_torch.TorchPreparedPanel(training),
        panels,
        {
            "training": training_info,
            "common_validation_overlap_removed": info["removed_validation_position_overlap"],
            "groups": receipts,
        },
    )


def run(
    directory,
    dataset,
    cache,
    weights,
    *,
    validation_dataset,
    validation_cache,
    groups,
    max_steps=30000,
    interval=1000,
    patience=7500,
    batch_size=64,
    learning_rate=1e-4,
    seed=20261026,
    train_cap=None,
    validation_cap=20000,
    wall_seconds=10800,
    stop_at=None,
    resume=None,
):
    if (
        min(max_steps, interval, patience, batch_size, wall_seconds) <= 0
        or not math.isfinite(wall_seconds)
    ):
        raise ValueError("positive training budgets required")
    if stop_at is not None and (type(stop_at) is not int or stop_at <= 0):
        raise ValueError("positive integer process boundary required")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    training_data = prepared_oracle.PreparedOracle(cache, source=dataset)
    validation_data = (
        training_data
        if (cache.resolve(), dataset.resolve())
        == (validation_cache.resolve(), validation_dataset.resolve())
        else prepared_oracle.PreparedOracle(validation_cache, source=validation_dataset)
    )
    training, panels, panel_info = common_panels(
        training_data,
        validation_data,
        groups,
        train_cap=train_cap,
        validation_cap=validation_cap,
        seed=seed,
    )
    learner_config = LearnerConfig(learning_rate=learning_rate)
    modules = (
        torch_network,
        packed_encoding,
        oracle_data,
        oracle_train,
        prepared_oracle,
        prepared_oracle_torch,
        torch_checkpoint,
        torch_learner,
    )
    config = {
        "task": "controlled-common-validation-soft-target",
        "schema": 1,
        "target_semantics": "native STM soft teacher WDL and policy; not self-learning",
        "initial_weights_sha256": torch_network.sha256(weights),
        "training_dataset_sha256": torch_network.sha256(dataset / "dataset.json"),
        "training_cache_sha256": torch_network.sha256(cache / "prepared.json"),
        "validation_dataset_sha256": torch_network.sha256(validation_dataset / "dataset.json"),
        "validation_cache_sha256": torch_network.sha256(validation_cache / "prepared.json"),
        "groups": [list(group) for group in groups],
        "panels": panel_info,
        "max_steps": max_steps,
        "interval": interval,
        "patience": patience,
        "batch_size": batch_size,
        "seed": seed,
        "learner": asdict(learner_config),
        "train_cap": train_cap,
        "validation_cap": validation_cap,
        "selection": "equal group mean policy_ce + value_ce; minimum improvement 0.0001",
        "frozen_prefixes": ["material_value_linear."],
        "code_sha256": {
            module.__name__: torch_network.sha256(Path(module.__file__)) for module in modules
        },
        "trainer_sha256": torch_network.sha256(Path(__file__)),
    }
    if resume is not None:
        learner, manifest, _ = torch_checkpoint.load_checkpoint(resume, expected_run_config=config)
        state = manifest["run_state"]
        if (
            manifest["step"] != state["cursor"]
            or resume.parent.parent.resolve() != directory.resolve()
        ):
            raise ValueError("resume requires original complete run and matching cursor")
    else:
        if directory.exists():
            raise FileExistsError("new controlled run requires a fresh directory")
        torch.manual_seed(seed)
        network = torch_network.load_weights(weights)
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(not name.startswith("material_value_linear."))
        learner = torch_learner.TorchLearner(network, config=learner_config)
        state = {
            "cursor": 0,
            "best_step": 0,
            "best_score": math.inf,
            "evaluations": [],
            "sample_trace_sha256": "0" * 64,
            "frozen_hashes": frozen_hashes(network),
        }
        directory.mkdir(parents=True)
        oracle_data.publish_json(
            directory / "metadata.json",
            {
                "source_commit": source,
                "config": config,
            },
        )
    inputs = set()
    for raw, packed in ((dataset, cache), (validation_dataset, validation_cache)):
        inputs.update((raw / "dataset.json", raw / "metadata.json", packed / "prepared.json"))
        inputs.update(raw.glob("*.json.gz"))
        inputs.update(packed.glob("*.npy"))

    def checkpoint():
        if frozen_hashes(learner.network) != state["frozen_hashes"]:
            raise ValueError("frozen auxiliary changed; checkpoint refused")
        state["cursor"] = learner.step
        path = directory / "checkpoints" / f"step-{learner.step:06d}"
        if not path.exists():
            torch_checkpoint.save_checkpoint(
                path,
                learner=learner,
                sampler=None,
                replay_paths=tuple(sorted(inputs)),
                run_state=state,
                run_config=config,
                source_commit=source,
            )

    def measure():
        measured = {name: oracle_train.evaluate(learner, panel) for name, panel in panels.items()}
        score = sum(value["total_ce"] for value in measured.values()) / len(measured)
        if not math.isfinite(score):
            raise ValueError("nonfinite validation; stop without selection")
        record = {"step": learner.step, "groups": measured, "selection_score": score}
        state["evaluations"].append(record)
        if score < state["best_score"] - 1e-4:
            state.update(best_step=learner.step, best_score=score)
        return record

    if not state["evaluations"]:
        measure()
    checkpoint()
    print(
        json.dumps(
            {"event": "ready", "panels": panel_info, "validation": state["evaluations"][-1]}
        ),
        flush=True,
    )
    stop = min(max_steps, max_steps if stop_at is None else stop_at)
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
            state["sample_trace_sha256"] = hashlib.sha256(
                bytes.fromhex(state["sample_trace_sha256"])
                + np.array(indices, dtype="<u8").tobytes()
            ).hexdigest()
            if learner.step % interval == 0:
                record = {
                    "validation": measure(),
                    "training_last_batch": asdict(metrics),
                    "best_step": state["best_step"],
                    "sample_trace_sha256": state["sample_trace_sha256"],
                    "wall_seconds": time.perf_counter() - started,
                }
                checkpoint()
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    if stop_at is not None and learner.step == stop_at and stop_at < max_steps:
        reason = "registered process boundary"
    checkpoint()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema": 1,
        "source_commit": source,
        "reason": reason,
        "status": "completed"
        if reason in ("maximum updates", "validation early stop")
        else "paused",
        "step": learner.step,
        "best_step": state["best_step"],
        "initial_validation": state["evaluations"][0],
        "best_validation": next(
            row for row in state["evaluations"] if row["step"] == state["best_step"]
        ),
        "best_weights_sha256": torch_network.sha256(
            directory / "checkpoints" / f"step-{state['best_step']:06d}" / "model.safetensors"
        ),
        "sample_trace_sha256": state["sample_trace_sha256"],
        "wall_seconds_this_process": time.perf_counter() - started,
        "self_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "self_maxrss_kib": usage.ru_maxrss,
        "resume_from": str(resume) if resume is not None else None,
        "promotion_ready": False,
    }
    oracle_data.publish_json(directory / f"result-step-{learner.step:06d}.json", result)
    return result


def frozen_hashes(network):
    return {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in network.named_parameters()
        if name.startswith("material_value_linear.")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "directory",
        "dataset",
        "cache",
        "weights",
        "validation_dataset",
        "validation_cache",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--group",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "MIN_FAMILY", "MAX_FAMILY"),
    )
    for name, default in (
        ("max-steps", 30000),
        ("interval", 1000),
        ("patience", 7500),
        ("batch-size", 64),
        ("seed", 20261026),
        ("validation-cap", 20000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--train-cap", type=int)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--wall-seconds", type=float, default=10800)
    args = vars(parser.parse_args())
    args["groups"] = tuple(
        (name, int(lower), int(upper)) for name, lower, upper in args.pop("group")
    )
    print(json.dumps(run(**args)), flush=True)


if __name__ == "__main__":
    main()
