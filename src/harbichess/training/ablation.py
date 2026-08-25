"""Train one fixed-compute continuation ablation from persisted replay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.shard import read_shard
from harbichess.replay.split import ReplaySplit
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.pilot import PilotConfig, run_sanity_pilot
from harbichess.training.resume import ResumeState


class ContinuationTreatment(StrEnum):
    OFF = "off"
    CURRENT = "current"
    FILTERED = "filtered"
    CONFIDENCE_GATED = "confidence_gated"
    REPETITION_RISK_GATED = "repetition_risk_gated"


@dataclass(frozen=True, slots=True)
class AblationConfig:
    ablation_id: str
    source_result: Path
    train_shard: Path
    validation_shard: Path
    treatment: ContinuationTreatment
    continuation_shards: tuple[Path, ...] = ()
    artifact_root: Path = Path("artifacts/ablations")
    steps: int = 200
    batch_size: int = 64
    seed: int = 2026082504
    continuation_fraction: float = 0.25
    reference_continuation_records: int | None = None
    recency_decay: float = 0.60

    def __post_init__(self) -> None:
        if not self.ablation_id or Path(self.ablation_id).name != self.ablation_id:
            raise ValueError("ablation_id must be one safe path segment")
        if self.steps <= 0 or self.batch_size <= 0 or self.seed < 0:
            raise ValueError("ablation steps and batch size must be positive")
        if not 0.0 <= self.continuation_fraction <= 1.0:
            raise ValueError("continuation fraction must be in [0, 1]")
        if not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency decay must be in (0, 1]")
        if self.treatment is ContinuationTreatment.OFF and self.continuation_shards:
            raise ValueError("continuation-off ablation cannot receive continuation shards")
        if self.treatment is not ContinuationTreatment.OFF and not self.continuation_shards:
            raise ValueError("continuation treatment requires at least one shard")
        if self.treatment is ContinuationTreatment.FILTERED and (
            self.reference_continuation_records is None or self.reference_continuation_records <= 0
        ):
            raise ValueError("filtered treatment requires the unfiltered reference size")


def matched_filtered_fraction(base_fraction: float, kept: int, total: int) -> float:
    """Keep expected exposures per retained record equal to the unfiltered arm."""

    if not 0.0 <= base_fraction <= 1.0 or kept <= 0 or total < kept:
        raise ValueError("filtered exposure inputs are invalid")
    return base_fraction * kept / total


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    config = payload["config"]
    return NetworkConfig(
        trunk_channels=int(config["trunk_channels"]),
        residual_blocks=int(config["residual_blocks"]),
        policy_channels=int(config["policy_channels"]),
        value_channels=int(config["value_channels"]),
        value_hidden=int(config["value_hidden"]),
    )


def run_ablation(config: AblationConfig) -> Path:
    output_root = config.artifact_root / config.ablation_id
    if output_root.exists():
        raise FileExistsError(f"ablation already exists: {output_root}")
    source = json.loads(config.source_result.read_text(encoding="utf-8"))
    baseline = source.get("baseline")
    if baseline is None:
        raise ValueError("ablation requires a persisted baseline")
    train = read_shard(config.train_shard)
    validation = read_shard(config.validation_shard)
    if train.header.split is not ReplaySplit.TRAIN:
        raise ValueError("ablation train shard must use the train split")
    if validation.header.split is not ReplaySplit.VALIDATION:
        raise ValueError("ablation validation shard must use the validation split")
    if {record.game_id for record in train.records} & {
        record.game_id for record in validation.records
    }:
        raise ValueError("ablation replay leaks games across train and validation")

    continuation_shards = tuple(read_shard(path) for path in config.continuation_shards)
    if any(shard.header.split is not ReplaySplit.TRAIN for shard in continuation_shards):
        raise ValueError("continuation ablation shards must use the train split")
    continuation = merge_continuation_replay(
        tuple(zip(config.continuation_shards, continuation_shards, strict=True)),
        recency_decay=config.recency_decay,
    )
    continuation_records = continuation.records
    fraction: float | None
    if config.treatment is ContinuationTreatment.OFF:
        fraction = None
    elif config.treatment is ContinuationTreatment.FILTERED:
        fraction = matched_filtered_fraction(
            config.continuation_fraction,
            len(continuation_records),
            config.reference_continuation_records,
        )
    else:
        fraction = config.continuation_fraction
    train_records = (*train.records, *continuation_records)

    network_config = _network_config(source)
    mx.random.seed(config.seed)
    network = HarbiChessNetwork(network_config)
    network.load_weights(str(baseline["path"]))
    source_config = source["config"]
    learner_config = LearnerConfig(
        learning_rate=float(source_config["learning_rate"]),
        weight_decay=1e-4,
        max_gradient_norm=5.0,
    )
    learner = MLXLearner(network, config=learner_config)
    train_evaluation = build_training_batch(train_records)
    validation_evaluation = build_training_batch(validation.records)
    started = time.perf_counter()
    report = run_sanity_pilot(
        learner,
        train_records,
        validation.records,
        config=PilotConfig(
            steps=config.steps,
            batch_size=config.batch_size,
            minimum_train_improvement=0.0,
            maximum_validation_ratio=2.0,
            validation_interval_steps=10,
            early_stopping_patience=config.steps + 1,
            minimum_validation_delta=1e-3,
            checkpoint_interval_steps=20,
            maximum_validation_checkpoints=4,
            continuation_fraction=fraction,
            continuation_game_weights=(continuation.game_weights if continuation_records else None),
            seed=config.seed,
        ),
        train_evaluation=train_evaluation,
        validation_evaluation=validation_evaluation,
    )
    elapsed = time.perf_counter() - started
    output_root.mkdir(parents=True)
    checkpoint_id = f"candidate-step-{report.steps:06d}"
    checkpoint_path = output_root / "checkpoints" / checkpoint_id
    sampler = GameBalancedSampler(
        train_records,
        seed=config.seed,
        continuation_fraction=fraction,
        continuation_game_weights=(continuation.game_weights if continuation_records else None),
    )
    sampler.set_rng_state(report.sampler_rng_state)
    commit = _source_commit()
    resume = ResumeState(
        schema_version=1,
        run_id=config.ablation_id,
        checkpoint_id=checkpoint_id,
        source_commit=commit,
        created_at=_now(),
        training_step=report.steps,
        lifetime_games=train.header.game_count,
        generation_games=train.header.game_count,
        training_elapsed_seconds=elapsed,
        replay_samples=len(train_records) + len(validation.records),
        replay_cursor=len(train_records) - 1,
        model_file="model.safetensors",
        optimizer_file="optimizer.safetensors",
        rng_file="sampler-rng.json",
    )
    saved = save_training_checkpoint(
        checkpoint_path,
        state=resume,
        learner=learner,
        sampler=sampler,
    )
    verification = MLXLearner(HarbiChessNetwork(network_config), config=learner_config)
    verification_sampler = GameBalancedSampler(
        train_records,
        seed=0,
        continuation_fraction=fraction,
        continuation_game_weights=(continuation.game_weights if continuation_records else None),
    )
    loaded = load_training_checkpoint(
        checkpoint_path,
        learner=verification,
        sampler=verification_sampler,
    )
    if loaded != saved:
        raise RuntimeError("ablation checkpoint verification failed")
    checkpoint = {
        "step": report.steps,
        "validation_loss": report.final_validation_loss,
        "path": str(checkpoint_path),
        "verified": True,
        "manifest": asdict(saved),
    }
    result_path = output_root / "result.json"
    _atomic_json(
        result_path,
        {
            "run_id": config.ablation_id,
            "source_commit": commit,
            "created_at": _now(),
            "passed": report.passed,
            "reasons": report.reasons,
            "treatment": config.treatment,
            "config": {
                **source_config,
                "run_id": config.ablation_id,
                "run_seed": config.seed,
                "training_steps": config.steps,
                "batch_size": config.batch_size,
                "continuation_batch_fraction": fraction or 0.0,
                "continuation_shards": [str(path) for path in config.continuation_shards],
            },
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "mlx_device": mx.device_info(),
            },
            "baseline": baseline,
            "checkpoint": checkpoint,
            "validation_checkpoints": [checkpoint],
            "timing": {"training_seconds": elapsed},
            "loss": {
                "attempted_steps": report.attempted_steps,
                "restored_step": report.steps,
                "initial_train": report.initial_train_loss,
                "final_train": report.final_train_loss,
                "initial_validation": report.initial_validation_loss,
                "final_validation": report.final_validation_loss,
                "best_validation": report.best_validation_loss,
                "maximum_gradient_norm": report.maximum_gradient_norm,
                "stop_reason": report.stop_reason,
            },
            "replay": {
                "fresh_train": asdict(train.header),
                "validation": asdict(validation.header),
                "continuation": [asdict(item) for item in continuation.sources],
                "continuation_records": len(continuation_records),
                "continuation_fraction": fraction or 0.0,
                "reference_continuation_records": config.reference_continuation_records,
            },
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--treatment", required=True, choices=tuple(ContinuationTreatment))
    parser.add_argument("--continuation-shard", action="append", type=Path, default=[])
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026082504)
    parser.add_argument("--continuation-fraction", type=float, default=0.25)
    parser.add_argument("--reference-continuation-records", type=int)
    parser.add_argument("--recency-decay", type=float, default=0.60)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_ablation(
        AblationConfig(
            ablation_id=arguments.ablation_id,
            source_result=arguments.source_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            treatment=ContinuationTreatment(arguments.treatment),
            continuation_shards=tuple(arguments.continuation_shard),
            artifact_root=arguments.artifact_root,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            continuation_fraction=arguments.continuation_fraction,
            reference_continuation_records=arguments.reference_continuation_records,
            recency_decay=arguments.recency_decay,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
