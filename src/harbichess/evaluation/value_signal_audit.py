"""Matched-exposure audit of replay value signal and game-level generalization."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json, _phase
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.training.batch import build_training_batch
from harbichess.training.full_gumbel_transfer import _network
from harbichess.training.joint_policy_value_transfer import (
    OutcomeGameBalancedSampler,
    _parameter_hash,
    _value_gate_reasons,
    _value_logits,
    _value_quality,
)
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.value_bootstrap import _freeze_to_value_head


@dataclass(frozen=True, slots=True)
class ValueSignalAuditConfig:
    output_dir: Path
    model_path: Path
    train_shard: Path
    validation_shard: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    expected_model_sha256: str = "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    learning_rate: float = 5e-4
    batch_size: int = 64
    steps: int = 140
    validation_interval: int = 20
    late_positions: int = 32
    seed: int = 2026083037

    def __post_init__(self) -> None:
        counts = (
            self.batch_size,
            self.steps,
            self.validation_interval,
            self.late_positions,
            self.seed,
        )
        if any(value <= 0 for value in counts) or self.steps % self.validation_interval:
            raise ValueError("value signal audit schedule is invalid")
        if self.learning_rate <= 0 or len(self.expected_model_sha256) != 64:
            raise ValueError("value signal audit optimizer or hash is invalid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    mean_left, mean_right = mean(left), mean(right)
    centered_left = tuple(value - mean_left for value in left)
    centered_right = tuple(value - mean_right for value in right)
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    return (
        sum(a * b for a, b in zip(centered_left, centered_right, strict=True)) / denominator
        if denominator
        else 0.0
    )


_PIECE_VALUES = {1: 1.0, 2: 3.0, 3: 3.25, 4: 5.0, 5: 9.0, 6: 0.0}


def _material_value(record: ReplayRecord, rules: PythonChessRules) -> float:
    board = rules.board(record.state)
    own = sum(
        len(board.pieces(piece_type, board.turn)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )
    opponent = sum(
        len(board.pieces(piece_type, not board.turn)) * value
        for piece_type, value in _PIECE_VALUES.items()
    )
    return math.tanh((own - opponent) / 39.0)


def _bucket(ply: int) -> str:
    if ply < 32:
        return "0-31"
    if ply < 64:
        return "32-63"
    if ply < 128:
        return "64-127"
    return "128+"


def _describe(records: tuple[ReplayRecord, ...], rules: PythonChessRules) -> dict[str, object]:
    known = tuple(record for record in records if record.outcome_value is not None)
    outcomes = tuple(float(record.outcome_value) for record in known)  # type: ignore[arg-type]
    materials = tuple(_material_value(record, rules) for record in known)
    roots = tuple(record.root_value for record in known)
    games = {record.game_id for record in known}
    return {
        "rows": len(records),
        "known_rows": len(known),
        "unknown_rows": len(records) - len(known),
        "known_games": len(games),
        "rows_per_known_game": len(known) / len(games),
        "outcomes": dict(Counter(str(record.outcome_value) for record in known)),
        "phases": dict(Counter(_phase(record.ply) for record in known)),
        "ply_buckets": dict(Counter(_bucket(record.ply) for record in known)),
        "material_outcome_pearson": _pearson(materials, outcomes),
        "stored_root_value_outcome_pearson": _pearson(roots, outcomes),
        "mean_absolute_stored_root_value": mean(abs(value) for value in roots),
    }


def _position_split(
    records: tuple[ReplayRecord, ...], *, seed: int
) -> tuple[tuple[ReplayRecord, ...], tuple[ReplayRecord, ...]]:
    grouped: dict[int, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        if record.outcome_value is None:
            raise ValueError("position split requires known outcomes")
        grouped[record.outcome_value].append(record)
    rng = random.Random(seed)
    train = []
    validation = []
    for outcome in (-1, 0, 1):
        rows = grouped[outcome]
        rng.shuffle(rows)
        boundary = round(len(rows) * 0.75)
        train.extend(rows[:boundary])
        validation.extend(rows[boundary:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return tuple(train), tuple(validation)


def _late(records: tuple[ReplayRecord, ...], *, count: int) -> tuple[ReplayRecord, ...]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        if record.outcome_value is not None:
            by_game[record.game_id].append(record)
    return tuple(
        record
        for game_id in sorted(by_game)
        for record in sorted(by_game[game_id], key=lambda row: row.ply)[-count:]
    )


def _shuffle_game_results(
    records: tuple[ReplayRecord, ...], *, seed: int
) -> tuple[ReplayRecord, ...]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        if record.outcome_value is not None:
            by_game[record.game_id].append(record)
    game_ids = tuple(sorted(by_game))
    results = []
    for game_id in game_ids:
        first = by_game[game_id][0]
        result = int(first.outcome_value) * (1 if first.side_to_move is Side.WHITE else -1)
        results.append(result)
    random.Random(seed).shuffle(results)
    shuffled = []
    for game_id, result in zip(game_ids, results, strict=True):
        for record in by_game[game_id]:
            outcome = (
                0 if result == 0 else result * (1 if record.side_to_move is Side.WHITE else -1)
            )
            shuffled.append(replace(record, outcome_value=outcome))
    return tuple(shuffled)


def _train_arm(
    label: str,
    train_records: tuple[ReplayRecord, ...],
    validation_records: tuple[ReplayRecord, ...],
    *,
    config: ValueSignalAuditConfig,
    store: SnapshotStore,
    snapshot,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    rules = PythonChessRules()
    train = MLXLearner.prepare_batch(build_training_batch(train_records, rules=rules))
    validation = MLXLearner.prepare_batch(build_training_batch(validation_records, rules=rules))
    train_outcomes = tuple(record.outcome_value for record in train_records)
    validation_outcomes = tuple(record.outcome_value for record in validation_records)
    network = _network()
    network.load_weights(str(config.model_path))
    frozen_before = _parameter_hash(
        network, excluded_prefixes=("value_conv.", "value_hidden.", "value_output.")
    )
    _freeze_to_value_head(network)
    learner = MLXLearner(
        network,
        config=LearnerConfig(
            learning_rate=config.learning_rate,
            weight_decay=0.0,
            policy_weight=0.0,
            value_weight=1.0,
        ),
    )
    baseline = _value_quality(_value_logits(network, validation), validation_outcomes)
    baseline_train = _value_quality(_value_logits(network, train), train_outcomes)
    best_macro = float(baseline["macro_cross_entropy"])
    best_step = 0
    best_snapshot = learner.snapshot()
    curve = [{"step": 0, "train": baseline_train, "validation": baseline}]
    sampler = OutcomeGameBalancedSampler(train_records, seed=config.seed)
    maximum_gradient_norm = 0.0
    for step in range(1, config.steps + 1):
        metrics = learner.train_step(train.select(sampler.sample_indices(config.batch_size)))
        maximum_gradient_norm = max(maximum_gradient_norm, metrics.unclipped_gradient_norm)
        if step % config.validation_interval:
            continue
        train_quality = _value_quality(_value_logits(network, train), train_outcomes)
        validation_quality = _value_quality(_value_logits(network, validation), validation_outcomes)
        curve.append({"step": step, "train": train_quality, "validation": validation_quality})
        macro = float(validation_quality["macro_cross_entropy"])
        if macro < best_macro:
            best_macro = macro
            best_step = step
            best_snapshot = learner.snapshot()
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"KRITIK value audit · {label} · {step}/{config.steps}",
            pilot_steps_completed=arm_index * config.steps + step,
            training_step=arm_index * config.steps + step,
            value_loss=float(validation_quality["cross_entropy"]),
        )
        store.write_atomic(snapshot)
    learner.restore(best_snapshot)
    selected = {
        "train": _value_quality(_value_logits(network, train), train_outcomes),
        "validation": _value_quality(_value_logits(network, validation), validation_outcomes),
    }
    reasons = _value_gate_reasons(
        baseline,
        selected["validation"],
        enforce_ece=False,
    )
    frozen_after = _parameter_hash(
        network, excluded_prefixes=("value_conv.", "value_hidden.", "value_output.")
    )
    if frozen_before != frozen_after:
        reasons = (*reasons, "value audit changed a frozen non-value parameter")
    return (
        {
            "label": label,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
            "train_games": len({record.game_id for record in train_records}),
            "validation_games": len({record.game_id for record in validation_records}),
            "overlapping_games": len(
                {record.game_id for record in train_records}
                & {record.game_id for record in validation_records}
            ),
            "baseline": baseline,
            "selected_step": best_step,
            "selected": selected,
            "maximum_gradient_norm": maximum_gradient_norm,
            "passed": not reasons,
            "reasons": reasons,
            "frozen_non_value_hash_before": frozen_before,
            "frozen_non_value_hash_after": frozen_after,
            "curve": curve,
        },
        snapshot,
    )


def _diagnosis(arms: dict[str, dict[str, object]]) -> dict[str, object]:
    passed = {label for label, arm in arms.items() if arm["passed"]}
    if "game-disjoint-shuffled" in passed:
        verdict = "leakage_or_spurious_game_identity"
    elif "game-disjoint-late32" in passed and "game-disjoint-all" not in passed:
        verdict = "early_ply_monte_carlo_variance"
    elif "position-split-all" in passed and "game-disjoint-all" not in passed:
        verdict = "insufficient_independent_games"
    elif not passed:
        verdict = "value_head_or_signal_not_learnable_at_fixed_exposure"
    else:
        verdict = "game_disjoint_value_signal_is_learnable"
    return {"verdict": verdict, "passed_arms": sorted(passed)}


def run_value_signal_audit(config: ValueSignalAuditConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"value signal audit output exists: {config.output_dir}")
    if _sha256(config.model_path) != config.expected_model_sha256:
        raise ValueError("value signal audit baseline checksum mismatch")
    rules = PythonChessRules()
    train_shard = read_shard(config.train_shard, rules=rules)
    validation_shard = read_shard(config.validation_shard, rules=rules)
    if min(train_shard.header.target_schema, validation_shard.header.target_schema) < 10:
        raise ValueError("value signal audit requires corrected max-ply targets")
    known_train = tuple(
        record for record in train_shard.records if record.outcome_value is not None
    )
    known_validation = tuple(
        record for record in validation_shard.records if record.outcome_value is not None
    )
    position_train, position_validation = _position_split(
        (*known_train, *known_validation), seed=config.seed
    )
    datasets = {
        "game-disjoint-all": (known_train, known_validation),
        "position-split-all": (position_train, position_validation),
        "game-disjoint-late32": (
            _late(known_train, count=config.late_positions),
            _late(known_validation, count=config.late_positions),
        ),
        "game-disjoint-shuffled": (
            _shuffle_game_results(known_train, seed=config.seed),
            known_validation,
        ),
    }
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="KRITIK value signal structural audit",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps * len(datasets),
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    arms = {}
    for index, (label, (train_records, validation_records)) in enumerate(datasets.items()):
        arms[label], snapshot = _train_arm(
            label,
            train_records,
            validation_records,
            config=config,
            store=store,
            snapshot=snapshot,
            arm_index=index,
        )
    diagnosis = _diagnosis(arms)
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
                    name: str(getattr(config, name))
                    for name in (
                        "output_dir",
                        "model_path",
                        "train_shard",
                        "validation_shard",
                        "telemetry_path",
                    )
                },
            },
            "descriptive": {
                "train": _describe(train_shard.records, rules),
                "validation": _describe(validation_shard.records, rules),
            },
            "arms": arms,
            "diagnosis": diagnosis,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = f"KRITIK value audit · {diagnosis['verdict']}"
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.FAILED,
        pilot_stop_reason="structural_value_audit",
        pilot_stop_detail=detail,
        pilot_reasons=(str(diagnosis["verdict"]),),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    print(
        run_value_signal_audit(
            ValueSignalAuditConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                train_shard=arguments.train_shard,
                validation_shard=arguments.validation_shard,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
