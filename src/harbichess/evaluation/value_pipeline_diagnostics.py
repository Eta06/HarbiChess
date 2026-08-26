"""Audit WDL calibration and checkpoint selection without starting a learner."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    _network_config,
    _source_commit,
)
from harbichess.replay.schema import TARGET_SCHEMA_VERSION, ReplayRecord
from harbichess.replay.shard import read_shard


@dataclass(frozen=True, slots=True)
class ValuePipelineDiagnosticConfig:
    run_result: Path
    shard: Path
    output_dir: Path
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("value diagnostic batch size must be positive")


def legacy_max_ply_game_ids(
    records: tuple[ReplayRecord, ...],
    *,
    max_plies: int,
    target_schema: int,
) -> frozenset[str]:
    """Identify legacy games whose artificial draw label is now masked by schema 10."""

    if target_schema >= TARGET_SCHEMA_VERSION:
        return frozenset()
    by_game: dict[str, list[ReplayRecord]] = {}
    for record in records:
        by_game.setdefault(record.game_id, []).append(record)
    return frozenset(
        game_id
        for game_id, game_records in by_game.items()
        if len(game_records) >= max_plies
        and max(record.ply for record in game_records) >= max_plies - 1
        and all(record.outcome_value == 0 for record in game_records)
    )


def _model_paths(run: dict[str, Any]) -> tuple[tuple[str, Path], ...]:
    models = [("baseline", Path(run["baseline"]["path"]))]
    models.extend(
        (
            f"validation-step-{candidate['step']}",
            Path(candidate["path"]) / "model.safetensors",
        )
        for candidate in run.get("validation_checkpoints", ())
    )
    return tuple(models)


def _model_metrics(
    network: HarbiChessNetwork,
    records: tuple[ReplayRecord, ...],
    *,
    excluded_games: frozenset[str],
    batch_size: int,
    rules: PythonChessRules,
) -> dict[str, Any]:
    encoder = BoardEncoder(rules)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    values: list[float] = []
    targets: list[int] = []
    predicted_classes: list[int] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        positions = tuple(encoder.encode(record.state) for record in chunk)
        shape = positions[0].shape
        inputs = mx.array([position.values for position in positions], dtype=mx.float32)
        inputs = inputs.reshape((len(positions), *shape))
        policy_logits, wdl_logits = network(inputs)
        policy_log_probs = policy_logits - mx.logsumexp(
            policy_logits,
            axis=1,
            keepdims=True,
        )
        wdl_log_probs = wdl_logits - mx.logsumexp(
            wdl_logits,
            axis=1,
            keepdims=True,
        )
        wdl_probs = mx.softmax(wdl_logits, axis=1)
        mx.eval(policy_log_probs, wdl_log_probs, wdl_probs)
        policy_rows = policy_log_probs.tolist()
        wdl_log_rows = wdl_log_probs.tolist()
        wdl_rows = wdl_probs.tolist()
        for index, record in enumerate(chunk):
            policy_losses.append(
                -sum(
                    probability * policy_rows[index][action]
                    for action, probability in record.policy
                )
            )
            if record.outcome_value is None or record.game_id in excluded_games:
                continue
            target_class = {1: 0, 0: 1, -1: 2}[record.outcome_value]
            row = wdl_rows[index]
            value_losses.append(-wdl_log_rows[index][target_class])
            values.append(row[0] - row[2])
            targets.append(record.outcome_value)
            predicted_classes.append(max(range(3), key=row.__getitem__))
    by_target = {
        str(target): mean(
            value
            for value, actual in zip(values, targets, strict=True)
            if actual == target
        )
        for target in sorted(set(targets))
    }
    return {
        "samples": len(records),
        "known_value_samples": len(targets),
        "policy_cross_entropy": mean(policy_losses),
        "value_cross_entropy": mean(value_losses),
        "combined_loss": mean(policy_losses) + mean(value_losses),
        "value_accuracy": mean(
            predicted == {1: 0, 0: 1, -1: 2}[target]
            for predicted, target in zip(predicted_classes, targets, strict=True)
        ),
        "expected_value_mean": mean(values),
        "expected_value_stddev": pstdev(values),
        "expected_value_mae": mean(
            abs(value - target) for value, target in zip(values, targets, strict=True)
        ),
        "expected_value_by_target": by_target,
    }


def run_value_pipeline_diagnostics(config: ValuePipelineDiagnosticConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    rules = PythonChessRules()
    shard = read_shard(config.shard, rules=rules)
    max_plies = int(run["config"]["max_plies"])
    excluded_games = legacy_max_ply_game_ids(
        shard.records,
        max_plies=max_plies,
        target_schema=shard.header.target_schema,
    )
    model_metrics: dict[str, dict[str, Any]] = {}
    for label, path in _model_paths(run):
        network = HarbiChessNetwork(_network_config(run))
        network.load_weights(str(path))
        network.eval()
        metrics = _model_metrics(
            network,
            shard.records,
            excluded_games=excluded_games,
            batch_size=config.batch_size,
            rules=rules,
        )
        model_metrics[label] = {"path": str(path), **metrics}

    baseline = model_metrics["baseline"]
    candidates = {key: value for key, value in model_metrics.items() if key != "baseline"}
    lowest_combined = min(model_metrics, key=lambda label: model_metrics[label]["combined_loss"])
    lowest_value = min(model_metrics, key=lambda label: model_metrics[label]["value_cross_entropy"])
    output = config.output_dir / "diagnostics.json"
    _atomic_json(
        output,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "replay": {
                "target_schema": shard.header.target_schema,
                "current_target_schema": TARGET_SCHEMA_VERSION,
                "samples": len(shard.records),
                "games": len({record.game_id for record in shard.records}),
                "raw_outcomes": dict(
                    Counter(str(record.outcome_value) for record in shard.records)
                ),
                "legacy_max_ply_games_excluded": len(excluded_games),
                "legacy_max_ply_samples_excluded": sum(
                    record.game_id in excluded_games for record in shard.records
                ),
            },
            "models": model_metrics,
            "selection_audit": {
                "lowest_combined_loss": lowest_combined,
                "lowest_value_loss": lowest_value,
                "combined_selection_masks_value_regression": (
                    lowest_combined != lowest_value
                    and model_metrics[lowest_combined]["value_cross_entropy"]
                    > baseline["value_cross_entropy"]
                ),
            },
            "gate": {
                "value_bottleneck_confirmed": (
                    baseline["expected_value_stddev"] < 0.02
                    and baseline["value_cross_entropy"] >= math.log(3) - 0.02
                ),
                "continuous_learner_authorized": False,
                "generation_authorized": False,
                "candidate_count": len(candidates),
            },
        },
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        run_value_pipeline_diagnostics(
            ValuePipelineDiagnosticConfig(
                run_result=arguments.run_result,
                shard=arguments.shard,
                output_dir=arguments.output_dir,
                batch_size=arguments.batch_size,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
