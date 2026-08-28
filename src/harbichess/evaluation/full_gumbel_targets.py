"""Generate provenance-complete Full Gumbel soft targets on frozen replay rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import action_to_legal_move, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


@dataclass(frozen=True, slots=True)
class FullGumbelTargetConfig:
    output_dir: Path
    model_path: Path
    train_shard: Path
    validation_shard: Path
    teacher_qualification_result: Path
    reference_target_result: Path | None = None
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    model_sha256: str = "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    train_positions: int = 384
    validation_positions: int = 192
    audit_positions: int = 32
    simulations: int = 256
    max_considered_actions: int = 16
    workers: int = 24
    inference_wait_seconds: float = 0.00025
    fixed_inference_batch_size: int = 24
    seed: int = 2026082879

    def __post_init__(self) -> None:
        counts = (
            self.train_positions,
            self.validation_positions,
            self.audit_positions,
            self.simulations,
            self.max_considered_actions,
            self.workers,
            self.fixed_inference_batch_size,
        )
        if any(value <= 0 for value in counts) or self.inference_wait_seconds < 0:
            raise ValueError("Full Gumbel target counts must be positive")
        if self.audit_positions > self.train_positions + self.validation_positions:
            raise ValueError("determinism audit exceeds selected target rows")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_key(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=16).digest()


def _identity(record: ReplayRecord) -> str:
    return f"{record.game_id}:{record.game_index}:{record.ply}"


def _stratum(record: ReplayRecord, rules: PythonChessRules) -> tuple[str, str, str]:
    phase = "opening" if record.ply < 20 else "middlegame" if record.ply < 80 else "endgame"
    board = rules.board(record.state)
    selected = action_to_legal_move(board, record.selected_action)
    tactical = (
        board.is_check()
        or board.is_capture(selected)
        or selected.promotion is not None
        or board.gives_check(selected)
    )
    outcome = (
        "unknown"
        if record.outcome_value is None
        else {-1: "losing", 0: "drawing", 1: "winning"}[record.outcome_value]
    )
    return phase, "tactical" if tactical else "quiet", outcome


def select_stratified_records(
    records: tuple[ReplayRecord, ...],
    *,
    count: int,
    seed: int,
    rules: PythonChessRules,
) -> tuple[ReplayRecord, ...]:
    """Round-robin composite strata and games with deterministic hashed ordering."""

    if count <= 0 or count > len(records):
        raise ValueError("stratified selection count must fit the replay")
    grouped: dict[tuple[str, str, str], dict[str, list[ReplayRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[_stratum(record, rules)][record.game_id].append(record)
    buckets: dict[tuple[str, str, str], deque[ReplayRecord]] = {}
    for stratum, games in grouped.items():
        ordered_games = sorted(games, key=lambda game: _hash_key(seed, f"game:{game}"))
        per_game = {
            game: deque(
                sorted(games[game], key=lambda row: _hash_key(seed, _identity(row)))
            )
            for game in ordered_games
        }
        rows: deque[ReplayRecord] = deque()
        while any(per_game.values()):
            for game in ordered_games:
                if per_game[game]:
                    rows.append(per_game[game].popleft())
        buckets[stratum] = rows
    strata = sorted(grouped, key=lambda key: _hash_key(seed, f"stratum:{key}"))
    selected = []
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError("stratified replay selection exhausted unexpectedly")
    return tuple(selected)


def _policy_metrics(
    raw: tuple[tuple[str, float], ...], target: tuple[tuple[str, float], ...]
) -> tuple[float, float, bool]:
    raw_map = dict(raw)
    target_map = dict(target)
    tv = 0.5 * sum(abs(target_map[move] - raw_map[move]) for move in raw_map)
    kl = sum(
        probability * math.log(probability / max(raw_map[move], 1e-300))
        for move, probability in target
        if probability > 0
    )
    raw_top = min(raw, key=lambda item: (-item[1], item[0]))[0]
    target_top = min(target, key=lambda item: (-item[1], item[0]))[0]
    return tv, kl, raw_top != target_top


def _target_row(
    record: ReplayRecord,
    search: FullGumbelMCTS,
    *,
    seed: int,
    rules: PythonChessRules,
) -> dict[str, object]:
    result = search.search(
        record.state,
        rng=random.Random(f"{seed}:{_identity(record)}"),
        add_root_noise=False,
    )
    board = rules.board(record.state)
    raw = tuple(sorted((move.uci, probability) for move, probability in result.network_priors))
    target = tuple(sorted((move.uci, probability) for move, probability in result.action_weights))
    actions = tuple(
        sorted(
            (move_to_action(board, board.parse_uci(move)), probability)
            for move, probability in target
        )
    )
    total = sum(probability for _, probability in target)
    legal = {move.uci() for move in board.legal_moves}
    if (
        not math.isclose(total, 1.0, abs_tol=1e-9)
        or any(not math.isfinite(probability) or probability < 0 for _, probability in target)
        or {move for move, _ in target} != legal
    ):
        raise RuntimeError("Full Gumbel emitted an invalid legal soft target")
    tv, kl, changed = _policy_metrics(raw, target)
    return {
        "identity": _identity(record),
        "game_id": record.game_id,
        "game_index": record.game_index,
        "ply": record.ply,
        "stratum": _stratum(record, rules),
        "outcome_value": record.outcome_value,
        "raw_policy": raw,
        "target": target,
        "action_target": actions,
        "selected_action": result.selected_action.uci,
        "root_value": result.root_value,
        "root_visits": tuple((move.move.uci, move.visits) for move in result.moves),
        "policy_tv": tv,
        "policy_kl": kl,
        "argmax_changed": changed,
    }


def _determinism_delta(first: dict[str, object], second: dict[str, object]) -> float:
    if (
        first["identity"] != second["identity"]
        or first["selected_action"] != second["selected_action"]
        or first["root_visits"] != second["root_visits"]
    ):
        return math.inf
    first_target = dict(first["target"])
    second_target = dict(second["target"])
    if first_target.keys() != second_target.keys():
        return math.inf
    return max(
        abs(float(first["root_value"]) - float(second["root_value"])),
        *(
            abs(float(first_target[move]) - float(second_target[move]))
            for move in first_target
        ),
    )


def _normalized_visits(value: object) -> tuple[tuple[str, int], ...]:
    return tuple((str(move), int(visits)) for move, visits in value)  # type: ignore[misc]


def _selection_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    strata: dict[str, int] = defaultdict(int)
    for row in rows:
        strata["/".join(row["stratum"])] += 1
    return {
        "positions": len(rows),
        "games": len({str(row["game_id"]) for row in rows}),
        "strata": tuple(sorted(strata.items())),
        "mean_policy_tv": sum(float(row["policy_tv"]) for row in rows) / len(rows),
        "mean_policy_kl": sum(float(row["policy_kl"]) for row in rows) / len(rows),
        "argmax_change_ratio": sum(bool(row["argmax_changed"]) for row in rows) / len(rows),
    }


def run_full_gumbel_targets(config: FullGumbelTargetConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"Full Gumbel target output exists: {config.output_dir}")
    qualification = json.loads(config.teacher_qualification_result.read_text(encoding="utf-8"))
    if not qualification.get("passed") or qualification.get("model_sha256") != config.model_sha256:
        raise ValueError("targets require the qualified Full Gumbel teacher and model")
    if _sha256(config.model_path) != config.model_sha256:
        raise ValueError("target model checksum mismatch")
    rules = PythonChessRules()
    train_shard = read_shard(config.train_shard, rules=rules)
    validation_shard = read_shard(config.validation_shard, rules=rules)
    if {record.game_id for record in train_shard.records} & {
        record.game_id for record in validation_shard.records
    }:
        raise ValueError("target train and validation games overlap")
    selected = {
        "train": select_stratified_records(
            train_shard.records,
            count=config.train_positions,
            seed=config.seed,
            rules=rules,
        ),
        "validation": select_stratified_records(
            validation_shard.records,
            count=config.validation_positions,
            seed=config.seed + 1,
            rules=rules,
        ),
    }
    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=16,
            residual_blocks=2,
            policy_channels=4,
            value_channels=2,
            value_hidden=32,
        )
    )
    network.load_weights(str(config.model_path))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(
            network, fixed_batch_size=config.fixed_inference_batch_size
        ),
        max_batch_size=min(config.fixed_inference_batch_size, config.workers),
        max_wait_seconds=config.inference_wait_seconds,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    search = FullGumbelMCTS(
        evaluator,
        rules=rules,
        config=FullGumbelConfig(
            simulations=config.simulations,
            max_considered_actions=config.max_considered_actions,
            gumbel_scale=0.0,
        ),
    )
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.EVALUATION,
        mode_detail="AKTAR Full Gumbel targets · 0/576",
        run_id=config.output_dir.name,
        active_games=config.train_positions + config.validation_positions,
        completed_games=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    rows: dict[str, tuple[dict[str, object], ...]] = {}
    completed = 0
    try:
        for partition, records in selected.items():
            def build(record: ReplayRecord) -> dict[str, object]:
                return _target_row(record, search, seed=config.seed, rules=rules)

            with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
                partition_rows = []
                for row in pool.map(build, records):
                    partition_rows.append(row)
                    completed += 1
                    if completed % 8 == 0 or completed == sum(map(len, selected.values())):
                        elapsed = time.perf_counter() - started
                        snapshot = replace(
                            snapshot,
                            updated_at=datetime.now(UTC).isoformat(),
                            mode_detail=(
                                f"AKTAR Full Gumbel targets · {completed}/"
                                f"{sum(map(len, selected.values()))}"
                            ),
                            active_games=sum(map(len, selected.values())) - completed,
                            completed_games=completed,
                            positions_per_second=batcher.statistics.positions / max(elapsed, 1e-9),
                        )
                        store.write_atomic(snapshot)
            rows[partition] = tuple(partition_rows)
        audit_source = tuple((*rows["train"], *rows["validation"]))[: config.audit_positions]
        audit_records = {
            _identity(record): record
            for records in selected.values()
            for record in records
        }
        audit_repeat = tuple(
            _target_row(
                audit_records[str(row["identity"])], search, seed=config.seed, rules=rules
            )
            for row in audit_source
        )
    finally:
        batcher.close()
    deltas = tuple(
        _determinism_delta(first, second)
        for first, second in zip(audit_source, audit_repeat, strict=True)
    )
    maximum_delta = max(deltas)
    elapsed = time.perf_counter() - started
    reference_equivalence: dict[str, object] | None = None
    reference_passed = True
    performance_passed = True
    if config.reference_target_result is not None:
        reference = json.loads(config.reference_target_result.read_text(encoding="utf-8"))
        reference_rows = {
            partition: {str(row["identity"]): row for row in reference["rows"][partition]}
            for partition in ("train", "validation")
        }
        action_mismatches = 0
        visit_mismatches = 0
        for partition, partition_rows in rows.items():
            for row in partition_rows:
                original = reference_rows[partition].get(str(row["identity"]))
                if original is None:
                    action_mismatches += 1
                    visit_mismatches += 1
                    continue
                action_mismatches += original["selected_action"] != row["selected_action"]
                visit_mismatches += _normalized_visits(
                    original["root_visits"]
                ) != _normalized_visits(row["root_visits"])
        reference_passed = action_mismatches == 0 and visit_mismatches == 0
        performance_passed = elapsed <= float(reference["elapsed_seconds"])
        reference_equivalence = {
            "artifact": str(config.reference_target_result),
            "selected_action_mismatches": action_mismatches,
            "root_visit_mismatches": visit_mismatches,
            "passed": reference_passed,
        }
    passed = maximum_delta <= 1e-12 and reference_passed and performance_passed
    result_path = config.output_dir / "result.json"
    config.output_dir.mkdir(parents=True)
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "algorithm": "full-gumbel-mctx-style-v1",
            "config": {
                **asdict(config),
                **{
                    name: str(getattr(config, name))
                    for name in (
                        "output_dir",
                        "model_path",
                        "train_shard",
                        "validation_shard",
                        "teacher_qualification_result",
                        "reference_target_result",
                        "telemetry_path",
                    )
                },
            },
            "provenance": {
                "model_sha256": _sha256(config.model_path),
                "train_shard_sha256": _sha256(config.train_shard),
                "validation_shard_sha256": _sha256(config.validation_shard),
                "train_payload_sha256": train_shard.header.payload_sha256,
                "validation_payload_sha256": validation_shard.header.payload_sha256,
                "teacher_qualification": str(config.teacher_qualification_result),
            },
            "selection": {
                partition: _selection_summary(partition_rows)
                for partition, partition_rows in rows.items()
            },
            "determinism": {
                "positions": len(deltas),
                "maximum_target_delta": maximum_delta,
                "tolerance": 1e-12,
                "passed": maximum_delta <= 1e-12,
            },
            "reference_equivalence": reference_equivalence,
            "performance_gate": {
                "reference_elapsed_seconds": (
                    None
                    if config.reference_target_result is None
                    else float(reference["elapsed_seconds"])
                ),
                "elapsed_seconds": elapsed,
                "passed": performance_passed,
            },
            "rows": rows,
            "elapsed_seconds": elapsed,
            "inference": {
                **asdict(batcher.statistics),
                "positions_per_second": batcher.statistics.positions / max(elapsed, 1e-9),
            },
            "passed": passed,
            "learner_transfer_authorized": passed,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "AKTAR targets qualified · learner transfer authorized"
            if passed
            else "AKTAR target provenance failed · learner blocked"
        ),
        active_games=0,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-shard", type=Path, required=True)
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--reference-target-result", type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--inference-wait-seconds", type=float, default=0.00025)
    parser.add_argument("--fixed-inference-batch-size", type=int, default=24)
    arguments = parser.parse_args(argv)
    result = run_full_gumbel_targets(
        FullGumbelTargetConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            teacher_qualification_result=arguments.teacher_qualification,
            reference_target_result=arguments.reference_target_result,
            telemetry_path=arguments.telemetry,
            inference_wait_seconds=arguments.inference_wait_seconds,
            fixed_inference_batch_size=arguments.fixed_inference_batch_size,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
