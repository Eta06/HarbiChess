"""Controlled fixed self-search distillation with full native resume, no oracle labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.nn import functional as F

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, sha256
from harbichess.core.network_config import NetworkConfig
from harbichess.replay.shard import read_shard
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch


def data_records(replay: Path) -> tuple[tuple, tuple, dict]:
    paths = sorted(replay.glob("*.gz"))
    records = tuple(r for p in paths for r in read_shard(p).records)
    training = tuple(r for r in records if r.game_index % 48 % 4 != 0)
    validation = tuple(r for r in records if r.game_index % 48 % 4 == 0)
    if not training or not validation:
        raise ValueError("empty family split")
    return training, validation, {str(p.resolve()): sha256(p) for p in paths}


def prepare(replay: Path, cache: Path) -> dict:
    training, validation, sources = data_records(replay)
    if cache.exists():
        raise FileExistsError(cache)
    # Only batch tensor conversion uses this network, never inference/training.
    network = TorchChessNetwork(NetworkConfig(trunk_channels=1, residual_blocks=1))
    learner = TorchLearner(network)
    panels = {}
    for name, records in (("train", training), ("validation", validation)):
        chunks = [
            learner.prepare_batch(build_training_batch(records[start : start + 256]))
            for start in range(0, len(records), 256)
        ]
        panels[name] = {
            field.name: torch.cat([getattr(b, field.name) for b in chunks])
            for field in fields(TorchTrainingBatch)
        }
        del chunks
    train_histories = {(r.root_fen, r.moves) for r in training}
    validation_histories = {(r.root_fen, r.moves) for r in validation}
    if train_histories & validation_histories:
        raise ValueError("complete history overlap between family splits")
    metadata = {
        "schema": 1,
        "sources": sources,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_games": len({r.game_id for r in training}),
        "validation_games": len({r.game_id for r in validation}),
        "known_outcome_rows": {
            "train": sum(r.outcome_value is not None for r in training),
            "validation": sum(r.outcome_value is not None for r in validation),
        },
        "full_history_overlap": 0,
        "semantics": "original terminal replay12; unknown masked; stored own-search policy",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"panels": panels, "metadata": metadata}, cache)
    metadata = {**metadata, "cache_sha256": sha256(cache)}
    cache.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def evaluate(learner: TorchLearner, panel: TorchTrainingBatch) -> dict:
    policy_sum, value_sum, known = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, panel.size, 64):
            b = panel.select(tuple(range(start, min(panel.size, start + 64))))
            policy, value = learner.network(b.inputs)
            policy_sum += float(
                -(b.policies * F.log_softmax(policy.masked_fill(~b.legal_masks, -1e9), 1)).sum()
            )
            value_sum += float(
                (F.cross_entropy(value, b.wdl, reduction="none") * b.value_weights).sum()
            )
            known += int(b.value_weights.sum())
    return {
        "policy_ce": policy_sum / panel.size,
        "value_ce": value_sum / known if known else None,
        "rows": panel.size,
        "known_value_rows": known,
    }


def frozen_digest(network) -> str:
    digest = hashlib.sha256()
    for key, value in network.named_parameters():
        if not value.requires_grad:
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261012)
    training_records, validation_records, sources = data_records(args.replay)
    metadata = json.loads(args.cache.with_suffix(".json").read_text())
    if metadata["sources"] != sources or metadata["cache_sha256"] != sha256(args.cache):
        raise ValueError("replay/cache source integrity mismatch")
    data = torch.load(args.cache, weights_only=True)
    training = TorchTrainingBatch(**data["panels"]["train"])
    validation = TorchTrainingBatch(**data["panels"]["validation"])
    if (training.size, validation.size) != (len(training_records), len(validation_records)):
        raise ValueError("cached sample order/count mismatch")
    sampler = GameBalancedSampler(training_records, seed=20261012)
    config = {
        "schema": 1,
        "task": "UFUK-fixed-self-policy-v1",
        "seed": 20261012,
        "initial_weights_sha256": sha256(args.weights),
        "cache_sha256": metadata["cache_sha256"],
        "max_steps": 8000,
        "interval": 250,
        "patience": 1500,
        "batch_size": 64,
        "split": "game_index%48%4==0 held out; all source generations",
    }
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.resume:
        learner, manifest, sampler_rng = load_checkpoint(args.resume, expected_run_config=config)
        sampler.set_rng_state(sampler_rng)
        state = manifest["run_state"]
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        network = load_weights(args.weights)
        for key, parameter in network.named_parameters():
            parameter.requires_grad_(key.startswith(("pair_", "policy_adapter_blocks.")))
        learner = TorchLearner(network, config=LearnerConfig(learning_rate=5e-5))
        baseline = evaluate(learner, validation)
        state = {
            "baseline": baseline,
            "best": baseline,
            "best_step": 0,
            "frozen_sha256": frozen_digest(network),
            "evaluations": [],
            "sample_trace_sha256": "0" * 64,
        }
        (args.output / "data-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    probe = validation.inputs[:8]
    initial = load_weights(args.weights)
    with torch.no_grad():
        reference_value = initial(probe)[1]
    reason = "maximum updates"

    def save():
        return save_checkpoint(
            args.output / f"checkpoints/step-{learner.step:06d}",
            learner=learner,
            sampler=sampler,
            replay_paths=tuple(Path(p) for p in sources),
            run_state=state,
            run_config=config,
            source_commit=source_commit,
        )

    if not args.resume:
        save()
    while learner.step < 8000:
        if args.stop_at is not None and learner.step >= args.stop_at:
            reason = "registered process boundary"
            break
        if time.perf_counter() - started >= args.wall_seconds:
            reason = "wall budget exhausted"
            break
        if learner.step - state["best_step"] >= 1500:
            reason = "validation early stop"
            break
        indices = sampler.sample_indices(64)
        metrics = learner.train_step(training.select(indices))
        state["sample_trace_sha256"] = hashlib.sha256(
            (state["sample_trace_sha256"] + json.dumps(indices)).encode()
        ).hexdigest()
        if learner.step % 250 == 0:
            if frozen_digest(learner.network) != state["frozen_sha256"]:
                raise ValueError("frozen inherited parameters changed")
            with torch.no_grad():
                if not torch.equal(reference_value, learner.network(probe)[1]):
                    raise ValueError("policy update changed frozen value function")
            measured = evaluate(learner, validation)
            if measured["policy_ce"] < state["best"]["policy_ce"]:
                state["best"], state["best_step"] = measured, learner.step
            state["evaluations"].append(
                {
                    "step": learner.step,
                    "validation": measured,
                    "last_training": asdict(metrics),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            save()
            print(json.dumps(state["evaluations"][-1]), flush=True)
    if not (args.output / f"checkpoints/step-{learner.step:06d}").exists():
        save()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "source_commit": source_commit,
        "status": "paused"
        if args.stop_at is not None
        else (
            "completed" if reason in ("maximum updates", "validation early stop") else "incomplete"
        ),
        "reason": reason,
        "step": learner.step,
        "state": state,
        "wall_seconds": time.perf_counter() - started,
        "parent_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
        "scope": (
            "fixed own-search policy learning; frozen value; "
            "not fresh closed-loop or played strength"
        ),
    }
    filename = (
        f"result-pause-{learner.step:06d}.json" if args.stop_at is not None else "result.json"
    )
    path = args.output / filename
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        print(json.dumps(prepare(args.replay, args.cache)))
        return
    if not args.output or not args.weights or args.wall_seconds <= 0:
        parser.error("output, weights and positive wall budget required")
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
