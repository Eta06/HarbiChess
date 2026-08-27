"""Frozen-head value bootstrap diagnostic; never starts self-play or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    _network_config,
    _source_commit,
)
from harbichess.evaluation.value_pipeline_diagnostics import legacy_max_ply_game_ids
from harbichess.replay.shard import ReplayShard, read_shard
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.learner import LearnerConfig, MLXLearner


@dataclass(frozen=True, slots=True)
class ValueBootstrapConfig:
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    learning_rates: tuple[float, ...] = (1e-4, 2e-4, 5e-4, 1e-3)
    steps: int = 1_000
    batch_size: int = 64
    validation_interval: int = 50
    seed: int = 2026082622

    def __post_init__(self) -> None:
        if (
            not self.learning_rates
            or any(rate <= 0 for rate in self.learning_rates)
            or self.steps <= 0
            or self.batch_size <= 0
            or self.validation_interval <= 0
            or self.steps % self.validation_interval
            or self.seed < 0
        ):
            raise ValueError("value bootstrap configuration is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corrected_records(
    shard: ReplayShard,
    *,
    max_plies: int,
) -> tuple[tuple, frozenset[str]]:
    excluded = legacy_max_ply_game_ids(
        shard.records,
        max_plies=max_plies,
        target_schema=shard.header.target_schema,
    )
    return (
        tuple(
            replace(record, outcome_value=None) if record.game_id in excluded else record
            for record in shard.records
        ),
        excluded,
    )


def _freeze_to_value_head(network: HarbiChessNetwork) -> None:
    network.freeze()
    network.value_conv.unfreeze()
    network.value_hidden.unfreeze()
    network.value_output.unfreeze()


def run_value_bootstrap(config: ValueBootstrapConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"bootstrap output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("value bootstrap requires a persisted baseline")
    rules = PythonChessRules()
    train_shard = read_shard(config.train_shard, rules=rules)
    validation_shard = read_shard(config.validation_shard, rules=rules)
    max_plies = int(run["config"]["max_plies"])
    train_records, excluded_train = _corrected_records(train_shard, max_plies=max_plies)
    validation_records, excluded_validation = _corrected_records(
        validation_shard,
        max_plies=max_plies,
    )
    train_batch = MLXLearner.prepare_batch(build_training_batch(train_records, rules=rules))
    validation_batch = MLXLearner.prepare_batch(
        build_training_batch(validation_records, rules=rules)
    )
    config.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    variants = []
    best_variant: tuple[float, int, str, Path] | None = None
    for rate in config.learning_rates:
        mx.random.seed(config.seed)
        network = HarbiChessNetwork(_network_config(run))
        network.load_weights(str(baseline["path"]))
        _freeze_to_value_head(network)
        learner = MLXLearner(
            network,
            config=LearnerConfig(
                learning_rate=rate,
                weight_decay=0.0,
                policy_weight=0.0,
                value_weight=1.0,
            ),
        )
        sampler = GameBalancedSampler(train_records, seed=config.seed)
        initial_total, initial_policy, initial_value = learner.evaluate_loss(validation_batch)
        best_value = initial_value
        best_step = 0
        best_snapshot = learner.snapshot()
        curve = []
        for step in range(1, config.steps + 1):
            learner.train_step(train_batch.select(sampler.sample_indices(config.batch_size)))
            if step % config.validation_interval == 0:
                total, policy, value = learner.evaluate_loss(validation_batch)
                curve.append(
                    {
                        "step": step,
                        "total_loss": total,
                        "policy_loss": policy,
                        "value_loss": value,
                    }
                )
                if value < best_value:
                    best_value = value
                    best_step = step
                    best_snapshot = learner.snapshot()
        learner.restore(best_snapshot)
        final_total, final_policy, final_value = learner.evaluate_loss(validation_batch)
        label = f"lr-{rate:.0e}"
        model_path = config.output_dir / "variants" / label / "model.safetensors"
        model_path.parent.mkdir(parents=True)
        network.save_weights(str(model_path))
        variants.append(
            {
                "label": label,
                "learning_rate": rate,
                "initial_total_loss": initial_total,
                "initial_policy_loss": initial_policy,
                "initial_value_loss": initial_value,
                "best_step": best_step,
                "best_value_loss": best_value,
                "final_total_loss": final_total,
                "final_policy_loss": final_policy,
                "final_value_loss": final_value,
                "policy_loss_change": final_policy - initial_policy,
                "model_path": str(model_path),
                "model_sha256": _sha256(model_path),
                "curve": curve,
            }
        )
        candidate = (best_value, best_step, label, model_path)
        if best_variant is None or candidate < best_variant:
            best_variant = candidate
    assert best_variant is not None
    result_path = config.output_dir / "bootstrap.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "baseline": baseline,
            "data_correction": {
                "train_legacy_max_ply_games_masked": len(excluded_train),
                "validation_legacy_max_ply_games_masked": len(excluded_validation),
            },
            "variants": variants,
            "selection": {
                "metric": "minimum validation WDL cross-entropy",
                "label": best_variant[2],
                "step": best_variant[1],
                "value_loss": best_variant[0],
                "model_path": str(best_variant[3]),
                "model_sha256": _sha256(best_variant[3]),
            },
            "controls": {
                "trunk_frozen": True,
                "policy_head_frozen": True,
                "policy_loss_weight": 0.0,
                "only_value_head_trainable": True,
            },
            "gate": {
                "continuous_learner_authorized": False,
                "generation_authorized": False,
                "note": "diagnostic bootstrap must pass teacher qualification",
            },
            "timing": {"elapsed_seconds": time.perf_counter() - started},
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--learning-rates", default="0.0001,0.0002,0.0005,0.001")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026082622)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        run_value_bootstrap(
            ValueBootstrapConfig(
                run_result=arguments.run_result,
                train_shard=arguments.train_shard,
                validation_shard=arguments.validation_shard,
                output_dir=arguments.output_dir,
                learning_rates=tuple(
                    float(value) for value in arguments.learning_rates.split(",") if value
                ),
                steps=arguments.steps,
                batch_size=arguments.batch_size,
                validation_interval=arguments.validation_interval,
                seed=arguments.seed,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
