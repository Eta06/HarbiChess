"""Qualify deterministic sequential halving against standard root PUCT."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.evaluation.consensus_target import _record_index, _record_policy, _top_actions
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    _interval,
    _network_config,
    _source_commit,
)
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.sequential_halving import deterministic_sequential_halving
from harbichess.search.value_oracle import (
    OracleValueEvaluator,
    ProcessTacticalOracle,
    TacticalOracleConfig,
)


@dataclass(frozen=True, slots=True)
class SequentialHalvingQualificationConfig:
    consistency_result: Path
    q_reliability_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    budgets: tuple[int, ...] = (512, 800)
    maximum_considered_actions: int = 16
    workers: int = 24
    oracle_workers: int = 8
    max_batch_size: int = 48
    max_wait_seconds: float = 0.00025
    oracle_depth: int = 1
    verifier_depth: int = 4
    bootstrap_samples: int = 2_000
    seed: int = 2026082821
    minimum_action_agreement: float = 0.75
    maximum_harmful_ratio: float = 0.10
    harmful_delta: float = -0.025
    maximum_verified_regret: float = 0.10
    maximum_delta_regression_vs_q: float = 0.01
    minimum_best_action_coverage: float = 0.80

    def __post_init__(self) -> None:
        if (
            len(self.budgets) != 2
            or tuple(sorted(set(self.budgets))) != self.budgets
            or min(
                *self.budgets,
                self.maximum_considered_actions,
                self.workers,
                self.oracle_workers,
                self.max_batch_size,
                self.oracle_depth,
                self.verifier_depth,
                self.bootstrap_samples,
                self.seed,
            )
            <= 0
            or self.maximum_considered_actions <= 1
            or self.verifier_depth <= self.oracle_depth
            or self.max_wait_seconds < 0
            or not 0 <= self.minimum_action_agreement <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.harmful_delta > 0
            or self.maximum_verified_regret < 0
            or self.maximum_delta_regression_vs_q < 0
            or not 0 <= self.minimum_best_action_coverage <= 1
        ):
            raise ValueError("sequential halving qualification configuration is invalid")


def _summary(
    rows: tuple[Mapping[str, object], ...],
    *,
    config: SequentialHalvingQualificationConfig,
    seed: int,
) -> dict[str, object]:
    low, high = map(str, config.budgets)
    deltas = tuple(float(row["budgets"][high]["verified_delta_vs_raw"]) for row in rows)
    return {
        "positions": len(rows),
        "selected_action_agreement": mean(
            row["budgets"][low]["selected_action"]
            == row["budgets"][high]["selected_action"]
            for row in rows
        ),
        "mean_verified_delta_vs_raw": mean(deltas),
        "verified_delta_95_interval": _interval(
            deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "harmful_count": sum(delta <= config.harmful_delta for delta in deltas),
        "harmful_ratio": sum(delta <= config.harmful_delta for delta in deltas) / len(rows),
        "mean_verified_regret": mean(
            float(row["budgets"][high]["verified_regret"]) for row in rows
        ),
        "best_action_coverage": mean(
            bool(row["budgets"][high]["considered_contains_best_verified"]) for row in rows
        ),
        "all_budgets_exact": all(
            int(row["budgets"][str(budget)]["evaluation_slots"]) == budget
            for row in rows
            for budget in config.budgets
        ),
        "mean_considered_actions": mean(
            int(row["budgets"][high]["considered_actions"]) for row in rows
        ),
    }


def _gate(
    summary: Mapping[str, object],
    *,
    standard_q_delta: float,
    config: SequentialHalvingQualificationConfig,
) -> dict[str, object]:
    reasons = []
    if float(summary["selected_action_agreement"]) < config.minimum_action_agreement:
        reasons.append("512-versus-800 selected-action agreement is below 75%")
    if float(summary["verified_delta_95_interval"][0]) <= 0:
        reasons.append("selected-action verified-improvement interval is not positive")
    if float(summary["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("harmful selected-action ratio exceeds 10%")
    if float(summary["mean_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("selected-action mean verified regret exceeds 0.10")
    if float(summary["mean_verified_delta_vs_raw"]) < (
        standard_q_delta - config.maximum_delta_regression_vs_q
    ):
        reasons.append("selected-action delta regresses standard top-Q by more than 0.01")
    if float(summary["best_action_coverage"]) < config.minimum_best_action_coverage:
        reasons.append("top-16 prior set best-action coverage is below 80%")
    if not summary["all_budgets_exact"]:
        reasons.append("sequential-halving evaluation-slot budget was not exact")
    return {"passed": not reasons, "reasons": reasons}


def run_sequential_halving_qualification(
    config: SequentialHalvingQualificationConfig,
) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"sequential halving output already exists: {config.output_dir}")
    consistency = json.loads(config.consistency_result.read_text(encoding="utf-8"))
    q_reliability = json.loads(config.q_reliability_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if q_reliability.get("gate", {}).get("passed"):
        raise ValueError("ODAK expects the failed TERAZI Q-target gate")
    rules = PythonChessRules()
    records = {
        "train": _record_index(read_shard(config.train_shard, rules=rules).records),
        "validation": _record_index(read_shard(config.validation_shard, rules=rules).records),
    }
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
    search = MCTS(
        teacher,
        rules=rules,
        config=SearchConfig(simulations=1, dirichlet_fraction=0.0),
    )
    started = time.perf_counter()
    output_rows: dict[str, tuple[dict[str, object], ...]] = {}
    try:
        for partition in ("train", "validation"):
            matched: list[tuple[Mapping[str, object], ReplayRecord]] = []
            for row in consistency["rows"][partition]:
                key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
                record = records[partition].get(key)
                if record is None:
                    raise ValueError(f"MIHENK row is absent from replay: {key}")
                matched.append((row, record))

            results = {}
            for budget in config.budgets:

                def inspect(
                    index: int,
                    budget: int = budget,
                    partition: str = partition,
                    matched: list[tuple[Mapping[str, object], ReplayRecord]] = matched,
                ):
                    return deterministic_sequential_halving(
                        search,
                        matched[index][1].state,
                        budget=budget,
                        rng=random.Random(f"{config.seed}:{partition}:{budget}:{index}"),
                        maximum_considered_actions=config.maximum_considered_actions,
                    )

                with ThreadPoolExecutor(max_workers=min(config.workers, len(matched))) as pool:
                    results[budget] = tuple(pool.map(inspect, range(len(matched))))

            work = []
            for index, (_, record) in enumerate(matched):
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
            for index, (source_row, record) in enumerate(matched):
                raw_top = _top_actions(_record_policy(record, rules), 1)[0]
                best_value = max(verified[index].values())
                budget_rows = {}
                for budget in config.budgets:
                    result = results[budget][index]
                    selected = result.selected_action.uci
                    considered = {move.uci for move in result.considered_actions}
                    budget_rows[str(budget)] = {
                        "selected_action": selected,
                        "selected_verified_value": verified[index][selected],
                        "verified_delta_vs_raw": (
                            verified[index][selected] - verified[index][raw_top]
                        ),
                        "verified_regret": best_value - verified[index][selected],
                        "considered_actions": len(considered),
                        "considered_contains_best_verified": any(
                            verified[index][action] == best_value for action in considered
                        ),
                        "evaluation_slots": result.evaluation_slots,
                        "rounds": result.rounds,
                        "action_values": tuple(
                            (move.uci, value) for move, value in result.action_values
                        ),
                        "action_slots": tuple(
                            (move.uci, slots) for move, slots in result.action_slots
                        ),
                    }
                rows.append(
                    {
                        "partition": partition,
                        "game_id": source_row["game_id"],
                        "game_index": source_row["game_index"],
                        "ply": source_row["ply"],
                        "raw_top_action": raw_top,
                        "budgets": budget_rows,
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
    standard_q_delta = float(
        q_reliability["summaries"]["validation"]["mean_top_q_verified_delta_vs_raw"]
    )
    gate = _gate(summaries["validation"], standard_q_delta=standard_q_delta, config=config)
    result_path = config.output_dir / "qualification.json"
    _atomic_json(
        result_path,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "consistency_result": str(config.consistency_result),
                "q_reliability_result": str(config.q_reliability_result),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "standard_q_validation_delta": standard_q_delta,
            "summaries": summaries,
            "gate": {
                **gate,
                "soft_target_experiment_authorized": gate["passed"],
                "learner_ablation_authorized": False,
                "arena_authorized": False,
                "generation_authorized": False,
                "promotion_authorized": False,
            },
            "rows": output_rows,
            "inference": asdict(batcher.statistics),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency-result", required=True, type=Path)
    parser.add_argument("--q-reliability-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    path = run_sequential_halving_qualification(
        SequentialHalvingQualificationConfig(
            consistency_result=arguments.consistency_result,
            q_reliability_result=arguments.q_reliability_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
