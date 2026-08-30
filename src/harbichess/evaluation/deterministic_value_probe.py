"""Frozen diagnostic for deterministic value representation learnability."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.schema import ReplayRecord
from harbichess.search.value_oracle import DeterministicTacticalOracle, TacticalOracleConfig
from harbichess.training.full_gumbel_transfer import _network, _snapshot
from harbichess.training.value_bootstrap import _freeze_to_value_head


@dataclass(frozen=True, slots=True)
class DeterministicValueProbeConfig:
    output_dir: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    maximum_train_positions: int = 8192
    maximum_validation_positions: int = 4096
    batch_size: int = 64
    steps: int = 200
    validation_interval: int = 20
    learning_rate: float = 5e-4
    seed: int = 2026083061

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_train_positions,
                self.maximum_validation_positions,
                self.batch_size,
                self.steps,
                self.validation_interval,
                self.seed,
            )
            <= 0
            or self.steps % self.validation_interval
            or self.learning_rate <= 0
        ):
            raise ValueError("deterministic value probe configuration is invalid")


def _round_robin(records: tuple[ReplayRecord, ...], maximum: int) -> tuple[ReplayRecord, ...]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        by_game[record.game_id].append(record)
    ordered = {
        game: sorted(rows, key=lambda row: row.ply) for game, rows in sorted(by_game.items())
    }
    selected = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for rows in ordered.values():
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _prepare(
    records: tuple[ReplayRecord, ...], rules: PythonChessRules
) -> tuple[mx.array, mx.array]:
    encoder = BoardEncoder(rules)
    oracle = DeterministicTacticalOracle(
        rules=rules, config=TacticalOracleConfig(depth=0)
    )
    positions = []
    targets = []
    for record in records:
        board = rules.board(record.state)
        positions.append(encoder.encode_state(record.state, board).values)
        targets.append(oracle.value(record.state))
    inputs = mx.array(positions, dtype=mx.float32).reshape(
        (len(positions), 8, 8, len(positions[0]) // 64)
    )
    values = mx.array(targets, dtype=mx.float32)
    mx.eval(inputs, values)
    return inputs, values


def _expected_score(network, inputs: mx.array) -> mx.array:
    _, logits = network(inputs)
    probabilities = mx.softmax(logits, axis=1)
    return probabilities[:, 0] - probabilities[:, 2]


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = mean(left), mean(right)
    x = [value - left_mean for value in left]
    y = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in x) * sum(value * value for value in y))
    return sum(a * b for a, b in zip(x, y, strict=True)) / denominator if denominator else 0.0


def _quality(network, inputs: mx.array, targets: mx.array) -> dict[str, float | int]:
    predictions = _expected_score(network, inputs)
    mx.eval(predictions)
    predicted = predictions.tolist()
    expected = targets.tolist()
    errors = [a - b for a, b in zip(predicted, expected, strict=True)]
    return {
        "positions": len(expected),
        "mse": mean(error * error for error in errors),
        "mae": mean(abs(error) for error in errors),
        "pearson": _pearson(predicted, expected),
    }


class _ProbeLearner:
    def __init__(self, network, *, learning_rate: float) -> None:
        self.network = network
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(network, inputs: mx.array, targets: mx.array) -> mx.array:
        predictions = _expected_score(network, inputs)
        return mx.mean(mx.square(predictions - targets))

    def step(self, inputs: mx.array, targets: mx.array) -> float:
        loss, gradients = self.loss_and_grad(self.network, inputs, targets)
        self.optimizer.update(self.network, gradients)
        mx.eval(loss, self.network.parameters(), self.optimizer.state)
        return float(loss.item())


def _run_arm(
    label: str,
    train: tuple[mx.array, mx.array],
    validation: tuple[mx.array, mx.array],
    *,
    config: DeterministicValueProbeConfig,
    head_only: bool,
    store: SnapshotStore,
    snapshot,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    network = _network()
    network.load_weights(str(config.model_path))
    if head_only:
        _freeze_to_value_head(network)
    else:
        network.unfreeze()
        network.policy_conv.freeze()
        network.policy_linear.freeze()
    learner = _ProbeLearner(network, learning_rate=config.learning_rate)
    baseline = _quality(network, *validation)
    best_mse = float(baseline["mse"])
    best_step = 0
    best_weights = _snapshot(network)
    curve = [{"step": 0, "validation": baseline}]
    rng = random.Random(config.seed)
    for step in range(1, config.steps + 1):
        indices = tuple(rng.randrange(train[0].shape[0]) for _ in range(config.batch_size))
        rows = mx.array(indices, dtype=mx.int32)
        loss = learner.step(mx.take(train[0], rows, axis=0), mx.take(train[1], rows, axis=0))
        if step % config.validation_interval:
            continue
        quality = _quality(network, *validation)
        curve.append({"step": step, "batch_loss": loss, "validation": quality})
        if float(quality["mse"]) < best_mse:
            best_mse = float(quality["mse"])
            best_step = step
            best_weights = _snapshot(network)
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"KRITIK deterministic value probe · {label} · {step}/{config.steps}",
            pilot_steps_completed=arm_index * config.steps + step,
            training_step=arm_index * config.steps + step,
            value_loss=float(quality["mse"]),
        )
        store.write_atomic(snapshot)
    network.load_weights(list(best_weights))
    selected = _quality(network, *validation)
    reasons = []
    if float(selected["mse"]) > float(baseline["mse"]) * 0.5:
        reasons.append("held-out deterministic-value MSE did not improve by 50 percent")
    if float(selected["pearson"]) < 0.80:
        reasons.append("held-out deterministic-value Pearson is below 0.80")
    if float(selected["mae"]) > 0.05:
        reasons.append("held-out deterministic-value MAE exceeds 0.05")
    return {
        "label": label,
        "baseline": baseline,
        "selected_step": best_step,
        "selected": selected,
        "passed": not reasons,
        "reasons": reasons,
        "curve": curve,
    }, snapshot


def run_deterministic_value_probe(config: DeterministicValueProbeConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"deterministic probe output exists: {config.output_dir}")
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    train_records, validation_records, split = _split_games(games, seed=pool_config.seed)
    train_records = _round_robin(train_records, config.maximum_train_positions)
    validation_records = _round_robin(
        validation_records, config.maximum_validation_positions
    )
    rules = PythonChessRules()
    train = _prepare(train_records, rules)
    validation = _prepare(validation_records, rules)
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="KRITIK deterministic value representation probe",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps * 2,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    arms = {}
    for index, (label, head_only) in enumerate(
        (("head-only", True), ("full-representation", False))
    ):
        arms[label], snapshot = _run_arm(
            label,
            train,
            validation,
            config=config,
            head_only=head_only,
            store=store,
            snapshot=snapshot,
            arm_index=index,
        )
    if arms["head-only"]["passed"]:
        verdict = "existing_trunk_value_features_qualified"
    elif arms["full-representation"]["passed"]:
        verdict = "shared_representation_learning_required"
    else:
        verdict = "deterministic_value_representation_failed"
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": {
                **asdict(config),
                **{
                    key: str(getattr(config, key))
                    for key in ("output_dir", "model_path", "runs_root", "telemetry_path")
                },
            },
            "provenance": provenance,
            "split": split,
            "selected_train_positions": len(train_records),
            "selected_validation_positions": len(validation_records),
            "arms": arms,
            "verdict": verdict,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = f"KRITIK deterministic value probe · {verdict}"
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=(
            PilotStatus.FAILED
            if verdict == "deterministic_value_representation_failed"
            else PilotStatus.PASSED
        ),
        pilot_stop_reason="deterministic_value_probe",
        pilot_stop_detail=detail,
        pilot_reasons=tuple(
            reason for arm in arms.values() for reason in arm["reasons"]
        ),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    print(
        run_deterministic_value_probe(
            DeterministicValueProbeConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
