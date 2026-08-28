"""Audit whether root child-Q is a reliable teacher signal independent of visits."""

from __future__ import annotations

import argparse
import json
import math
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
from harbichess.search.mcts import MCTS, SearchConfig, SearchResult
from harbichess.search.value_oracle import (
    OracleValueEvaluator,
    ProcessTacticalOracle,
    TacticalOracleConfig,
)


@dataclass(frozen=True, slots=True)
class SearchQReliabilityConfig:
    consistency_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    budgets: tuple[int, ...] = (512, 800)
    workers: int = 24
    oracle_workers: int = 8
    max_batch_size: int = 48
    max_wait_seconds: float = 0.00025
    oracle_depth: int = 1
    verifier_depth: int = 4
    bootstrap_samples: int = 2_000
    seed: int = 2026082820
    minimum_q_verified_correlation: float = 0.35
    maximum_harmful_ratio: float = 0.10
    harmful_delta: float = -0.025
    maximum_verified_regret: float = 0.10
    minimum_top_q_agreement: float = 0.75
    minimum_cross_budget_q_correlation: float = 0.70
    maximum_delta_regression_vs_visits: float = 0.01

    def __post_init__(self) -> None:
        if (
            len(self.budgets) != 2
            or tuple(sorted(set(self.budgets))) != self.budgets
            or min(
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
            or not -1 <= self.minimum_q_verified_correlation <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.harmful_delta > 0
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_top_q_agreement <= 1
            or not -1 <= self.minimum_cross_budget_q_correlation <= 1
            or self.maximum_delta_regression_vs_visits < 0
        ):
            raise ValueError("search Q reliability configuration is invalid")


def _ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for action, _ in ordered[start:end]:
            ranks[action] = rank
        start = end
    return ranks


def _spearman(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    actions = first.keys() & second.keys()
    if len(actions) < 2:
        return 0.0
    left = _ranks({action: first[action] for action in actions})
    right = _ranks({action: second[action] for action in actions})
    left_mean = mean(left.values())
    right_mean = mean(right.values())
    numerator = sum(
        (left[action] - left_mean) * (right[action] - right_mean) for action in actions
    )
    left_scale = sum((left[action] - left_mean) ** 2 for action in actions)
    right_scale = sum((right[action] - right_mean) ** 2 for action in actions)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else 0.0


def _argmax(values: Mapping[str, float]) -> str:
    return min(values, key=lambda action: (-values[action], action))


def _search_metrics(
    result: SearchResult,
    verified: Mapping[str, float],
    *,
    raw_top: str,
) -> dict[str, object]:
    visited = {row.move.uci: row for row in result.moves if row.visits > 0}
    q = {action: row.mean_value for action, row in visited.items()}
    visits = {action: float(row.visits) for action, row in visited.items()}
    top_q = _argmax(q)
    top_visit = min(visited, key=lambda action: (-visited[action].visits, action))
    best_verified = max(verified.values())
    total_visits = sum(visits.values())
    return {
        "visited_actions": len(visited),
        "visited_fraction": len(visited) / len(verified),
        "q_verified_spearman": _spearman(q, verified),
        "visit_verified_spearman": _spearman(visits, verified),
        "visit_weighted_q_calibration_mae": sum(
            visits[action] / total_visits * abs(q[action] - verified[action]) for action in visited
        ),
        "top_q_action": top_q,
        "top_visit_action": top_visit,
        "top_q_verified_delta_vs_raw": verified[top_q] - verified[raw_top],
        "top_visit_verified_delta_vs_raw": verified[top_visit] - verified[raw_top],
        "top_q_verified_regret": best_verified - verified[top_q],
        "top_visit_verified_regret": best_verified - verified[top_visit],
        "q": tuple(sorted(q.items())),
        "visits": tuple(sorted((action, int(value)) for action, value in visits.items())),
    }


def _summary(
    rows: tuple[Mapping[str, object], ...], *, config: SearchQReliabilityConfig, seed: int
) -> dict[str, object]:
    high = str(config.budgets[-1])
    low = str(config.budgets[0])
    q_deltas = tuple(float(row["budgets"][high]["top_q_verified_delta_vs_raw"]) for row in rows)
    visit_deltas = tuple(
        float(row["budgets"][high]["top_visit_verified_delta_vs_raw"]) for row in rows
    )
    return {
        "positions": len(rows),
        "mean_high_budget_q_verified_spearman": mean(
            float(row["budgets"][high]["q_verified_spearman"]) for row in rows
        ),
        "mean_high_budget_visit_verified_spearman": mean(
            float(row["budgets"][high]["visit_verified_spearman"]) for row in rows
        ),
        "mean_top_q_verified_delta_vs_raw": mean(q_deltas),
        "top_q_verified_delta_95_interval": _interval(
            q_deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "mean_top_visit_verified_delta_vs_raw": mean(visit_deltas),
        "top_visit_verified_delta_95_interval": _interval(
            visit_deltas, samples=config.bootstrap_samples, seed=seed + 1
        ),
        "top_q_harmful_count": sum(delta <= config.harmful_delta for delta in q_deltas),
        "top_q_harmful_ratio": sum(delta <= config.harmful_delta for delta in q_deltas)
        / len(rows),
        "mean_top_q_verified_regret": mean(
            float(row["budgets"][high]["top_q_verified_regret"]) for row in rows
        ),
        "mean_top_visit_verified_regret": mean(
            float(row["budgets"][high]["top_visit_verified_regret"]) for row in rows
        ),
        "top_q_agreement": mean(
            row["budgets"][low]["top_q_action"] == row["budgets"][high]["top_q_action"]
            for row in rows
        ),
        "mean_cross_budget_q_spearman": mean(
            float(row["cross_budget"]["q_spearman"]) for row in rows
        ),
        "mean_cross_budget_q_drift": mean(
            float(row["cross_budget"]["mean_absolute_q_drift"]) for row in rows
        ),
        "mean_high_budget_visited_fraction": mean(
            float(row["budgets"][high]["visited_fraction"]) for row in rows
        ),
        "mean_high_budget_q_calibration_mae": mean(
            float(row["budgets"][high]["visit_weighted_q_calibration_mae"]) for row in rows
        ),
    }


def _gate(summary: Mapping[str, object], config: SearchQReliabilityConfig) -> dict[str, object]:
    reasons = []
    if float(summary["mean_high_budget_q_verified_spearman"]) < (
        config.minimum_q_verified_correlation
    ):
        reasons.append("800-budget Q/verified correlation is below 0.35")
    if float(summary["top_q_verified_delta_95_interval"][0]) <= 0:
        reasons.append("top-Q verified-improvement interval is not positive")
    if float(summary["top_q_harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("top-Q harmful-action ratio exceeds 10%")
    if float(summary["mean_top_q_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("top-Q mean verified regret exceeds 0.10")
    if float(summary["top_q_agreement"]) < config.minimum_top_q_agreement:
        reasons.append("512-versus-800 top-Q agreement is below 75%")
    if float(summary["mean_cross_budget_q_spearman"]) < (
        config.minimum_cross_budget_q_correlation
    ):
        reasons.append("cross-budget Q correlation is below 0.70")
    if float(summary["mean_top_q_verified_delta_vs_raw"]) < (
        float(summary["mean_top_visit_verified_delta_vs_raw"])
        - config.maximum_delta_regression_vs_visits
    ):
        reasons.append("top-Q verified delta regresses top-visit by more than 0.01")
    return {"passed": not reasons, "reasons": reasons}


def run_search_q_reliability(config: SearchQReliabilityConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"search Q output already exists: {config.output_dir}")
    consistency = json.loads(config.consistency_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
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
            results: dict[int, tuple[SearchResult, ...]] = {}
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
                    matched: list[tuple[Mapping[str, object], ReplayRecord]] = matched,
                    partition: str = partition,
                ):
                    return search.search(
                        matched[index][1].state,
                        rng=random.Random(f"{config.seed}:{partition}:{budget}:{index}"),
                        add_root_noise=False,
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
                raw = _record_policy(record, rules)
                raw_top = _top_actions(raw, 1)[0]
                budget_metrics = {
                    str(budget): _search_metrics(
                        results[budget][index], verified[index], raw_top=raw_top
                    )
                    for budget in config.budgets
                }
                low = budget_metrics[str(config.budgets[0])]
                high = budget_metrics[str(config.budgets[1])]
                low_q = dict(low["q"])
                high_q = dict(high["q"])
                common = low_q.keys() & high_q.keys()
                rows.append(
                    {
                        "partition": partition,
                        "game_id": source_row["game_id"],
                        "game_index": source_row["game_index"],
                        "ply": source_row["ply"],
                        "raw_top_action": raw_top,
                        "budgets": budget_metrics,
                        "cross_budget": {
                            "common_visited_actions": len(common),
                            "q_spearman": _spearman(low_q, high_q),
                            "mean_absolute_q_drift": mean(
                                abs(low_q[action] - high_q[action]) for action in common
                            ),
                        },
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
    result_path = config.output_dir / "q-reliability.json"
    _atomic_json(
        result_path,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "consistency_result": str(config.consistency_result),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "q_target_authorized": gate["passed"],
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
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    path = run_search_q_reliability(
        SearchQReliabilityConfig(
            consistency_result=arguments.consistency_result,
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
