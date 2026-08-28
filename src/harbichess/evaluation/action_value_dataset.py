"""Generate and qualify a fresh non-overlapping action-value supervision set."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.consensus_target import _record_policy, _top_actions
from harbichess.evaluation.search_q_reliability import _spearman
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    _interval,
    _network_config,
    _source_commit,
    select_stratified_records,
)
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig, SearchResult
from harbichess.search.value_oracle import (
    OracleValueEvaluator,
    ProcessTacticalOracle,
    TacticalOracleConfig,
)


@dataclass(frozen=True, slots=True)
class ActionValueDatasetConfig:
    excluded_q_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    train_positions: int = 96
    validation_positions: int = 48
    budgets: tuple[int, ...] = (512, 800)
    workers: int = 24
    oracle_workers: int = 8
    max_batch_size: int = 48
    max_wait_seconds: float = 0.00025
    oracle_depth: int = 1
    verifier_depth: int = 4
    bootstrap_samples: int = 2_000
    seed: int = 2026082824
    minimum_q_verified_spearman: float = 0.35
    minimum_cross_budget_q_spearman: float = 0.70
    maximum_cross_budget_q_drift: float = 0.03
    minimum_top_two_overlap: float = 0.75
    maximum_harmful_ratio: float = 0.10
    harmful_delta: float = -0.025
    maximum_verified_regret: float = 0.10
    additional_excluded_q_results: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.budgets) != 2
            or tuple(sorted(set(self.budgets))) != self.budgets
            or min(
                self.train_positions,
                self.validation_positions,
                *self.budgets,
                self.workers,
                self.oracle_workers,
                self.max_batch_size,
                self.oracle_depth,
                self.verifier_depth,
                self.bootstrap_samples,
                self.seed,
            )
            <= 0
            or self.verifier_depth <= self.oracle_depth
            or self.max_wait_seconds < 0
            or not -1 <= self.minimum_q_verified_spearman <= 1
            or not -1 <= self.minimum_cross_budget_q_spearman <= 1
            or self.maximum_cross_budget_q_drift < 0
            or not 0 <= self.minimum_top_two_overlap <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.harmful_delta > 0
            or self.maximum_verified_regret < 0
        ):
            raise ValueError("action-value dataset configuration is invalid")


def _identity(record: ReplayRecord) -> tuple[str, int, int]:
    return record.game_id, record.game_index, record.ply


def _excluded_identities(payload: Mapping[str, object], partition: str):
    return {
        (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
        for row in payload["rows"][partition]
    }


def _search_payload(result: SearchResult) -> dict[str, object]:
    visited = tuple(row for row in result.moves if row.visits > 0)
    return {
        "q": tuple((row.move.uci, row.mean_value) for row in visited),
        "visits": tuple((row.move.uci, row.visits) for row in visited),
    }


def _top(values: Mapping[str, float], count: int = 1) -> tuple[str, ...]:
    return tuple(
        action for action, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:count]
    )


def _summary(
    rows: tuple[Mapping[str, object], ...], *, config: ActionValueDatasetConfig, seed: int
) -> dict[str, object]:
    deltas = tuple(float(row["top_q_verified_delta_vs_raw"]) for row in rows)
    return {
        "positions": len(rows),
        "mean_high_q_verified_spearman": mean(
            float(row["high_q_verified_spearman"]) for row in rows
        ),
        "mean_cross_budget_q_spearman": mean(float(row["cross_budget_q_spearman"]) for row in rows),
        "mean_cross_budget_q_drift": mean(float(row["cross_budget_q_drift"]) for row in rows),
        "mean_top_two_q_overlap": mean(float(row["top_two_q_overlap"]) for row in rows),
        "mean_top_q_verified_delta_vs_raw": mean(deltas),
        "top_q_verified_delta_95_interval": _interval(
            deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "top_q_harmful_count": sum(delta <= config.harmful_delta for delta in deltas),
        "top_q_harmful_ratio": sum(delta <= config.harmful_delta for delta in deltas) / len(rows),
        "mean_top_q_verified_regret": mean(float(row["top_q_verified_regret"]) for row in rows),
    }


def _gate(summary: Mapping[str, object], config: ActionValueDatasetConfig) -> dict[str, object]:
    reasons = []
    if float(summary["mean_high_q_verified_spearman"]) < config.minimum_q_verified_spearman:
        reasons.append("800-Q verified Spearman is below 0.35")
    if float(summary["mean_cross_budget_q_spearman"]) < (config.minimum_cross_budget_q_spearman):
        reasons.append("cross-budget Q Spearman is below 0.70")
    if float(summary["mean_cross_budget_q_drift"]) > config.maximum_cross_budget_q_drift:
        reasons.append("cross-budget Q drift exceeds 0.03")
    if float(summary["mean_top_two_q_overlap"]) < config.minimum_top_two_overlap:
        reasons.append("top-two Q-set overlap is below 75%")
    if float(summary["top_q_verified_delta_95_interval"][0]) <= 0:
        reasons.append("top-Q verified-improvement interval is not positive")
    if float(summary["top_q_harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("top-Q harmful-action ratio exceeds 10%")
    if float(summary["mean_top_q_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("top-Q verified regret exceeds 0.10")
    return {"passed": not reasons, "reasons": reasons}


def run_action_value_dataset(config: ActionValueDatasetConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"action-value dataset output exists: {config.output_dir}")
    excluded_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (config.excluded_q_result, *config.additional_excluded_q_results)
    ]
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    rules = PythonChessRules()
    all_records = {
        "train": read_shard(config.train_shard, rules=rules).records,
        "validation": read_shard(config.validation_shard, rules=rules).records,
    }
    selected = {}
    for index, partition in enumerate(("train", "validation")):
        forbidden = set().union(
            *(_excluded_identities(payload, partition) for payload in excluded_payloads)
        )
        available = tuple(
            record for record in all_records[partition] if _identity(record) not in forbidden
        )
        count = config.train_positions if partition == "train" else config.validation_positions
        selected[partition] = select_stratified_records(
            available, rules=rules, count=count, seed=config.seed + index
        )
    if {_identity(row) for row in selected["train"]} & {
        _identity(row) for row in selected["validation"]
    }:
        raise ValueError("fresh action-value train/validation identities overlap")

    store = SnapshotStore(config.telemetry_path)
    dashboard = store.read()
    total_searches = 2 * sum(map(len, selected.values()))
    completed_searches = 0
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.EVALUATION,
        mode_detail=f"Action-value teacher audit · 0/{total_searches}",
        pilot_status=PilotStatus.REPLAY,
        pilot_steps_planned=total_searches,
        pilot_steps_completed=0,
        pilot_steps_attempted=0,
        pilot_stop_reason="teacher_audit_running",
        pilot_stop_detail="Clean 512/800 search and independent verification",
        promotion_ready=False,
    )
    store.write_atomic(dashboard)

    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(run["baseline"]["path"]))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=config.max_batch_size,
        max_wait_seconds=config.max_wait_seconds,
    )
    neural = NeuralPositionEvaluator(batcher, rules=rules)
    teacher_oracle = ProcessTacticalOracle(
        TacticalOracleConfig(depth=config.oracle_depth), workers=config.oracle_workers
    )
    verifier = ProcessTacticalOracle(
        TacticalOracleConfig(depth=config.verifier_depth), workers=config.oracle_workers
    )
    teacher = OracleValueEvaluator(neural, teacher_oracle)
    started = time.perf_counter()
    output_rows: dict[str, tuple[dict[str, object], ...]] = {}
    try:
        for partition, records in selected.items():
            results = {}
            for budget in config.budgets:
                search = MCTS(
                    teacher,
                    rules=rules,
                    config=SearchConfig(simulations=budget, dirichlet_fraction=0.0),
                )

                def inspect(
                    index: int,
                    search: MCTS = search,
                    budget: int = budget,
                    partition: str = partition,
                    records: tuple[ReplayRecord, ...] = records,
                ):
                    return search.search(
                        records[index].state,
                        rng=random.Random(f"{config.seed}:{partition}:{budget}:{index}"),
                        add_root_noise=False,
                    )

                with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
                    results[budget] = tuple(pool.map(inspect, range(len(records))))
                completed_searches += len(records)
                dashboard = replace(
                    dashboard,
                    updated_at=datetime.now(UTC).isoformat(),
                    mode_detail=(
                        f"Action-value teacher audit · {completed_searches}/{total_searches}"
                    ),
                    pilot_steps_completed=completed_searches,
                    pilot_steps_attempted=completed_searches,
                )
                store.write_atomic(dashboard)

            work = []
            for index, record in enumerate(records):
                board = rules.board(record.state)
                for action, _ in record.raw_policy:
                    move = action_to_legal_move(board, action).uci()
                    child = rules.apply(record.state, ChessMove(move))
                    work.append((index, move, child))

            def verify(item: tuple[int, str, object]):
                index, action, child = item
                return index, action, -verifier.value(child)

            with ThreadPoolExecutor(max_workers=config.oracle_workers) as pool:
                verified_items = tuple(pool.map(verify, work))
            verified: dict[int, dict[str, float]] = {}
            for index, action, value in verified_items:
                verified.setdefault(index, {})[action] = value

            rows = []
            low_budget, high_budget = config.budgets
            for index, record in enumerate(records):
                low = _search_payload(results[low_budget][index])
                high = _search_payload(results[high_budget][index])
                low_q = dict(low["q"])
                high_q = dict(high["q"])
                common = low_q.keys() & high_q.keys()
                raw_top = _top_actions(_record_policy(record, rules), 1)[0]
                top_q = _top(high_q)[0]
                best = max(verified[index].values())
                rows.append(
                    {
                        "partition": partition,
                        "game_id": record.game_id,
                        "game_index": record.game_index,
                        "ply": record.ply,
                        "budgets": {str(low_budget): low, str(high_budget): high},
                        "high_q_verified_spearman": _spearman(high_q, verified[index]),
                        "cross_budget_q_spearman": _spearman(low_q, high_q),
                        "cross_budget_q_drift": mean(
                            abs(low_q[action] - high_q[action]) for action in common
                        ),
                        "top_two_q_overlap": len(set(_top(low_q, 2)) & set(_top(high_q, 2))) / 2,
                        "top_q_action": top_q,
                        "top_q_verified_delta_vs_raw": (
                            verified[index][top_q] - verified[index][raw_top]
                        ),
                        "top_q_verified_regret": best - verified[index][top_q],
                        "verified_values": tuple(sorted(verified[index].items())),
                    }
                )
            output_rows[partition] = tuple(rows)
    finally:
        teacher_oracle.close()
        verifier.close()
        batcher.close()

    summaries = {
        partition: _summary(rows, config=config, seed=config.seed + index * 10)
        for index, (partition, rows) in enumerate(output_rows.items())
    }
    gate = _gate(summaries["validation"], config)
    result_path = config.output_dir / "dataset.json"
    _atomic_json(
        result_path,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "excluded_q_result": str(config.excluded_q_result),
                "additional_excluded_q_results": tuple(
                    map(str, config.additional_excluded_q_results)
                ),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
                "telemetry_path": str(config.telemetry_path),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "spatial_transfer_authorized": gate["passed"],
                "arena_authorized": False,
                "generation_authorized": False,
                "promotion_authorized": False,
            },
            "rows": output_rows,
            "inference": asdict(batcher.statistics),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "Action-value teacher audit passed"
            if gate["passed"]
            else "Action-value teacher audit failed"
        ),
        teacher_qualification_status="passed" if gate["passed"] else "failed",
        teacher_qualification_positions=sum(map(len, output_rows.values())),
        teacher_qualification_result=str(result_path),
        pilot_status=PilotStatus.PASSED if gate["passed"] else PilotStatus.FAILED,
        pilot_steps_completed=total_searches,
        pilot_steps_attempted=total_searches,
        pilot_stop_reason="teacher_audit_complete",
        pilot_stop_detail=(
            "Raw Q teacher passed frozen gates" if gate["passed"] else "; ".join(gate["reasons"])
        ),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excluded-q-result", required=True, type=Path)
    parser.add_argument("--additional-excluded-q-result", action="append", default=[], type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--seed", type=int, default=2026082824)
    parser.add_argument("--train-positions", type=int, default=96)
    parser.add_argument("--validation-positions", type=int, default=48)
    arguments = parser.parse_args(argv)
    path = run_action_value_dataset(
        ActionValueDatasetConfig(
            excluded_q_result=arguments.excluded_q_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
            additional_excluded_q_results=tuple(arguments.additional_excluded_q_result),
            seed=arguments.seed,
            train_positions=arguments.train_positions,
            validation_positions=arguments.validation_positions,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
