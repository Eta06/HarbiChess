"""Audit continuation targets against a fixed champion MCTS search."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig, SearchResult


class AuditVerdict(StrEnum):
    RELIABLE = "reliable"
    UNCERTAIN = "uncertain"
    HARMFUL = "harmful"


@dataclass(frozen=True, slots=True)
class AuditThresholds:
    value_margin: float = 0.05
    minimum_target_visits: int = 4
    minimum_target_visit_mass: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.value_margin) or self.value_margin < 0:
            raise ValueError("audit value margin must be finite and non-negative")
        if self.minimum_target_visits <= 0:
            raise ValueError("minimum target visits must be positive")
        if not 0.0 <= self.minimum_target_visit_mass <= 1.0:
            raise ValueError("minimum target visit mass must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ContinuationAudit:
    game_id: str
    source_run: str
    verdict: AuditVerdict
    outcome_value: int
    recorded_root_value: float
    champion_root_value: float
    root_value_delta: float
    repeat_moves: tuple[str, ...]
    repeat_visits: int
    repeat_visit_mass: float
    repeat_mcts_value: float
    target_moves: int
    target_visits: int
    target_visit_mass: float
    target_mcts_value: float | None
    target_repeat_overlap: int
    champion_selected_move: str
    champion_selected_repeats: bool
    target_contains_champion_selection: bool


def _weighted_value(statistics) -> float | None:
    visits = sum(item.visits for item in statistics)
    if visits <= 0:
        return None
    return sum(item.mean_value * item.visits for item in statistics) / visits


def audit_continuation_record(
    record: ReplayRecord,
    search: SearchResult,
    rules: PythonChessRules,
    *,
    source_run: str,
    thresholds: AuditThresholds | None = None,
) -> ContinuationAudit:
    settings = thresholds or AuditThresholds()
    if not search.moves or search.simulations <= 0:
        raise ValueError("continuation audit requires a non-terminal MCTS result")
    board = rules.board(record.state)
    action_by_move = {
        item.move: move_to_action(board, board.parse_uci(item.move.uci)) for item in search.moves
    }
    target_actions = {action for action, _ in record.policy}
    repeating_moves = rules.claimable_threefold_moves(
        record.state,
        tuple(item.move for item in search.moves),
    )
    repeat = tuple(item for item in search.moves if item.move in repeating_moves)
    if not repeat:
        raise ValueError(
            f"continuation root no longer has a claimable repetition: {record.game_id}"
        )
    target = tuple(item for item in search.moves if action_by_move[item.move] in target_actions)
    total_visits = sum(item.visits for item in search.moves)
    repeat_visits = sum(item.visits for item in repeat)
    target_visits = sum(item.visits for item in target)
    repeat_value = _weighted_value(repeat)
    target_value = _weighted_value(target)
    if repeat_value is None:
        repeat_value = 0.0
    overlap = sum(item.move in repeating_moves for item in target)
    selected = search.select_move(temperature=0.0, rng=random.Random(0))
    target_mass = target_visits / total_visits
    selected_repeats = selected in repeating_moves
    selected_in_target = action_by_move[selected] in target_actions
    if (
        overlap
        or (target_value is not None and target_value < repeat_value - settings.value_margin)
        or (not selected_repeats and not selected_in_target)
    ):
        verdict = AuditVerdict.HARMFUL
    elif (
        target_value is not None
        and not selected_repeats
        and selected_in_target
        and target_value >= repeat_value - min(settings.value_margin, 0.02)
        and target_visits >= settings.minimum_target_visits
        and target_mass >= settings.minimum_target_visit_mass
    ):
        verdict = AuditVerdict.RELIABLE
    else:
        verdict = AuditVerdict.UNCERTAIN
    return ContinuationAudit(
        game_id=record.game_id,
        source_run=source_run,
        verdict=verdict,
        outcome_value=record.outcome_value,
        recorded_root_value=record.root_value,
        champion_root_value=search.root_value,
        root_value_delta=search.root_value - record.root_value,
        repeat_moves=tuple(sorted(move.uci for move in repeating_moves)),
        repeat_visits=repeat_visits,
        repeat_visit_mass=repeat_visits / total_visits,
        repeat_mcts_value=repeat_value,
        target_moves=len(target),
        target_visits=target_visits,
        target_visit_mass=target_mass,
        target_mcts_value=target_value,
        target_repeat_overlap=overlap,
        champion_selected_move=selected.uci,
        champion_selected_repeats=selected_repeats,
        target_contains_champion_selection=selected_in_target,
    )


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


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    config = payload["config"]
    return NetworkConfig(
        trunk_channels=int(config["trunk_channels"]),
        residual_blocks=int(config["residual_blocks"]),
        policy_channels=int(config["policy_channels"]),
        value_channels=int(config["value_channels"]),
        value_hidden=int(config["value_hidden"]),
    )


def run_audit(
    *,
    run_result: Path,
    shard_paths: tuple[Path, ...],
    output_dir: Path,
    simulations: int,
    workers: int,
    thresholds: AuditThresholds,
) -> Path:
    if simulations <= 0 or workers <= 0 or not shard_paths:
        raise ValueError("audit simulations, workers, and shards must be non-empty")
    run = json.loads(run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("continuation audit requires a persisted champion baseline")
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(baseline["path"]))
    rules = PythonChessRules()
    shards = tuple((path, read_shard(path, rules=rules)) for path in shard_paths)
    indexed = tuple(
        (record, shard.header.run_id) for _, shard in shards for record in shard.records
    )
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, workers * 2),
        max_wait_seconds=0.00025,
    )
    search = MCTS(
        NeuralPositionEvaluator(batcher, rules=rules),
        rules=rules,
        config=SearchConfig(simulations=simulations, dirichlet_fraction=0.0),
    )
    started = time.perf_counter()

    def inspect(item: tuple[ReplayRecord, str]) -> tuple[ReplayRecord, ContinuationAudit]:
        record, source_run = item
        result = search.search(record.state, rng=random.Random(0), add_root_noise=False)
        return record, audit_continuation_record(
            record, result, rules, source_run=source_run, thresholds=thresholds
        )

    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(indexed))) as pool:
            inspected = tuple(pool.map(inspect, indexed))
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    statistics = batcher.statistics
    reliable = tuple(
        record for record, audit in inspected if audit.verdict is AuditVerdict.RELIABLE
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    filtered_path = output_dir / "continuation-reliable.jsonl.gz"
    filtered_header = None
    if reliable:
        filtered_header = write_shard_atomic(
            filtered_path,
            reliable,
            ShardMetadata(
                run_id=output_dir.name,
                generation=max(shard.header.generation for _, shard in shards) + 1,
                source_checkpoint=baseline["checkpoint_id"],
                source_commit=_source_commit(),
                created_at=datetime.now(UTC).isoformat(),
                split=ReplaySplit.TRAIN,
            ),
        )
    audits = tuple(audit for _, audit in inspected)
    counts = Counter(audit.verdict for audit in audits)
    source_counts = {
        source: dict(Counter(audit.verdict for audit in audits if audit.source_run == source))
        for source in sorted({audit.source_run for audit in audits})
    }
    result_path = output_dir / "audit.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "run_result": str(run_result),
            "champion": baseline,
            "config": {
                "simulations": simulations,
                "workers": workers,
                "thresholds": asdict(thresholds),
                "shards": [str(path) for path in shard_paths],
            },
            "summary": {
                "records": len(audits),
                "verdicts": dict(counts),
                "by_source": source_counts,
                "elapsed_seconds": elapsed,
                "filtered_records": len(reliable),
            },
            "inference": {
                **asdict(statistics),
                "average_batch_size": statistics.average_batch_size,
                "average_queue_wait_ms": statistics.average_queue_wait_ms,
            },
            "filtered_shard": (
                {"path": str(filtered_path), "header": asdict(filtered_header)}
                if filtered_header is not None
                else None
            ),
            "records": [asdict(audit) for audit in audits],
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--value-margin", type=float, default=0.05)
    parser.add_argument("--minimum-target-visits", type=int, default=4)
    parser.add_argument("--minimum-target-visit-mass", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_audit(
        run_result=arguments.run_result,
        shard_paths=tuple(arguments.shard),
        output_dir=arguments.output_dir,
        simulations=arguments.simulations,
        workers=arguments.workers,
        thresholds=AuditThresholds(
            value_margin=arguments.value_margin,
            minimum_target_visits=arguments.minimum_target_visits,
            minimum_target_visit_mass=arguments.minimum_target_visit_mass,
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
