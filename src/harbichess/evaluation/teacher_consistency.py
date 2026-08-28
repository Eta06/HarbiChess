"""Cross-budget clean-search consistency and verified-improvement audit."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
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
from harbichess.search.targets import visit_policy
from harbichess.search.value_oracle import (
    OracleValueEvaluator,
    ProcessTacticalOracle,
    TacticalOracleConfig,
)

Policy = tuple[tuple[ChessMove, float], ...]


@dataclass(frozen=True, slots=True)
class TeacherConsistencyConfig:
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    budgets: tuple[int, ...] = (64, 128, 256, 512, 800)
    train_positions: int = 96
    validation_positions: int = 48
    workers: int = 24
    oracle_workers: int = 8
    max_batch_size: int = 48
    max_wait_seconds: float = 0.00025
    oracle_depth: int = 1
    verifier_depth: int = 4
    bootstrap_samples: int = 2_000
    seed: int = 2026082816
    maximum_high_budget_tv: float = 0.25
    minimum_normalized_visit_margin: float = 0.03
    minimum_stable_verified_delta: float = 0.03
    maximum_harmful_verified_delta: float = -0.025
    minimum_stable_ratio: float = 0.20
    maximum_harmful_ratio: float = 0.10
    minimum_high_budget_agreement: float = 0.75

    def __post_init__(self) -> None:
        if (
            len(self.budgets) < 3
            or tuple(sorted(set(self.budgets))) != self.budgets
            or min(self.budgets) <= 0
            or min(
                self.train_positions,
                self.validation_positions,
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
            or not 0 <= self.maximum_high_budget_tv <= 1
            or not 0 <= self.minimum_normalized_visit_margin <= 1
            or not 0 <= self.minimum_stable_ratio <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or not 0 <= self.minimum_high_budget_agreement <= 1
        ):
            raise ValueError("teacher consistency configuration is invalid")


def _argmax(policy: Policy) -> ChessMove:
    return min(policy, key=lambda item: (-item[1], item[0].uci))[0]


def _probabilities(policy: Policy) -> dict[ChessMove, float]:
    return dict(policy)


def _tv(first: Policy, second: Policy) -> float:
    left, right = _probabilities(first), _probabilities(second)
    return 0.5 * sum(
        abs(left.get(move, 0.0) - right.get(move, 0.0)) for move in left.keys() | right.keys()
    )


def _kl(first: Policy, second: Policy) -> float:
    reference = _probabilities(second)
    return sum(
        probability * math.log(probability / max(reference.get(move, 0.0), 1e-12))
        for move, probability in first
        if probability > 0
    )


def _mixture(first: Policy, second: Policy) -> Policy:
    left, right = _probabilities(first), _probabilities(second)
    return tuple(
        sorted(
            (
                (move, 0.5 * (left.get(move, 0.0) + right.get(move, 0.0)))
                for move in left.keys() | right.keys()
            ),
            key=lambda item: item[0].uci,
        )
    )


def _policy_comparison(first: Policy, second: Policy) -> dict[str, float | bool]:
    mixture = _mixture(first, second)
    return {
        "top_action_agreement": _argmax(first) == _argmax(second),
        "tv": _tv(first, second),
        "forward_kl": _kl(first, second),
        "reverse_kl": _kl(second, first),
        "jensen_shannon": 0.5 * _kl(first, mixture) + 0.5 * _kl(second, mixture),
    }


def _stored_policy(record: ReplayRecord) -> Policy:
    board = PythonChessRules().board(record.state)
    return tuple(
        (ChessMove(action_to_legal_move(board, action).uci()), probability)
        for action, probability in record.policy
    )


def _search_row(result: SearchResult) -> dict[str, object]:
    policy = visit_policy(result)
    leader, runner = result.moves[:2]
    return {
        "policy": policy,
        "top_action": _argmax(policy),
        "leader_visits": leader.visits,
        "runner_visits": runner.visits,
        "leader_visit_share": leader.visits / result.simulations,
        "normalized_visit_margin": (leader.visits - runner.visits) / result.simulations,
        "q_margin": leader.mean_value - runner.mean_value,
        "root_value": result.root_value,
    }


def _classify_row(
    budget_rows: dict[int, dict[str, object]],
    *,
    verified_delta: float,
    config: TeacherConsistencyConfig,
) -> tuple[str, tuple[str, ...]]:
    high_budgets = config.budgets[-3:]
    top_actions = tuple(budget_rows[budget]["top_action"] for budget in config.budgets)
    high_policies = tuple(budget_rows[budget]["policy"] for budget in high_budgets)
    maximum_tv = max(_tv(first, second) for first, second in combinations(high_policies, 2))
    minimum_margin = min(
        float(budget_rows[budget]["normalized_visit_margin"]) for budget in high_budgets
    )
    if verified_delta <= config.maximum_harmful_verified_delta:
        return "harmful", ("high-budget action is independently worse than raw",)
    reasons = []
    if len(set(top_actions)) != 1:
        reasons.append("top action changes across budgets")
    if maximum_tv > config.maximum_high_budget_tv:
        reasons.append("high-budget policy TV exceeds limit")
    if minimum_margin < config.minimum_normalized_visit_margin:
        reasons.append("high-budget normalized visit margin is too small")
    if verified_delta < config.minimum_stable_verified_delta:
        reasons.append("verified improvement is below stable minimum")
    return (
        ("stable/high-confidence", ())
        if not reasons
        else ("budget-sensitive/ambiguous", tuple(reasons))
    )


def _summary(rows: tuple[dict[str, object], ...], *, config: TeacherConsistencyConfig, seed: int):
    classes = tuple(str(row["classification"]) for row in rows)
    stable_deltas = tuple(
        float(row["verified_delta_vs_raw"])
        for row in rows
        if row["classification"] == "stable/high-confidence"
    )
    all_deltas = tuple(float(row["verified_delta_vs_raw"]) for row in rows)
    high_first, high_second = config.budgets[-2:]
    return {
        "positions": len(rows),
        "stable_count": classes.count("stable/high-confidence"),
        "stable_ratio": classes.count("stable/high-confidence") / len(rows),
        "ambiguous_count": classes.count("budget-sensitive/ambiguous"),
        "ambiguous_ratio": classes.count("budget-sensitive/ambiguous") / len(rows),
        "harmful_count": classes.count("harmful"),
        "harmful_ratio": classes.count("harmful") / len(rows),
        "high_budget_top_action_agreement": mean(
            row["budgets"][str(high_first)]["top_action"]
            == row["budgets"][str(high_second)]["top_action"]
            for row in rows
        ),
        "stable_verified_delta_mean": mean(stable_deltas) if stable_deltas else 0.0,
        "stable_verified_delta_95_interval": _interval(
            stable_deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "all_verified_delta_mean": mean(all_deltas),
        "all_verified_delta_95_interval": _interval(
            all_deltas, samples=config.bootstrap_samples, seed=seed + 1
        ),
        "mean_maximum_high_budget_tv": mean(
            float(row["maximum_high_budget_tv"]) for row in rows
        ),
        "mean_minimum_high_budget_visit_margin": mean(
            float(row["minimum_high_budget_visit_margin"]) for row in rows
        ),
    }


def _gate(summary: dict[str, object], config: TeacherConsistencyConfig) -> dict[str, object]:
    reasons = []
    if float(summary["stable_ratio"]) < config.minimum_stable_ratio:
        reasons.append("stable target ratio is below 20%")
    if float(summary["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("harmful target ratio exceeds 10%")
    if float(summary["stable_verified_delta_95_interval"][0]) <= 0:
        reasons.append("stable verified-improvement interval is not positive")
    if float(summary["high_budget_top_action_agreement"]) < (
        config.minimum_high_budget_agreement
    ):
        reasons.append("512-versus-800 top-action agreement is below 75%")
    if float(summary["all_verified_delta_95_interval"][0]) < 0:
        reasons.append("full 800-budget verified-improvement interval is negative")
    return {"passed": not reasons, "reasons": reasons}


def run_teacher_consistency(config: TeacherConsistencyConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"teacher consistency output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if run.get("mode") != "generation_only" or not run.get("passed"):
        raise ValueError("teacher consistency requires the qualified generation replay")
    rules = PythonChessRules()
    partitions = {
        "train": select_stratified_records(
            read_shard(config.train_shard, rules=rules).records,
            rules=rules,
            count=config.train_positions,
            seed=config.seed,
        ),
        "validation": select_stratified_records(
            read_shard(config.validation_shard, rules=rules).records,
            rules=rules,
            count=config.validation_positions,
            seed=config.seed + 1,
        ),
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
        for partition, records in partitions.items():
            partition_name = partition
            selected_records = records
            with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
                raw_evaluations = tuple(
                    pool.map(lambda record: neural.evaluate(record.state), selected_records)
                )
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
                    selected_records: tuple[ReplayRecord, ...] = selected_records,
                    partition_name: str = partition_name,
                ):
                    return search.search(
                        selected_records[index].state,
                        rng=random.Random(
                            f"{config.seed}:{partition_name}:{budget}:{index}"
                        ),
                        add_root_noise=False,
                    )

                with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
                    results[budget] = tuple(pool.map(inspect, range(len(records))))

            budget_rows = tuple(
                {budget: _search_row(results[budget][index]) for budget in config.budgets}
                for index in range(len(records))
            )
            actions = {
                (index, _argmax(raw_evaluations[index].priors))
                for index in range(len(records))
            } | {
                (index, row[config.budgets[-1]]["top_action"])
                for index, row in enumerate(budget_rows)
            }

            def verify(
                item: tuple[int, ChessMove],
                selected_records: tuple[ReplayRecord, ...] = selected_records,
            ):
                index, move = item
                child = rules.apply(selected_records[index].state, move)
                return item, -verifier.value(child)

            with ThreadPoolExecutor(max_workers=min(config.workers, len(actions))) as pool:
                verified = dict(
                    pool.map(
                        verify,
                        sorted(actions, key=lambda item: (item[0], item[1].uci)),
                    )
                )

            rows = []
            for index, (record, raw, by_budget) in enumerate(
                zip(records, raw_evaluations, budget_rows, strict=True)
            ):
                raw_action = _argmax(raw.priors)
                final_action = by_budget[config.budgets[-1]]["top_action"]
                delta = verified[(index, final_action)] - verified[(index, raw_action)]
                classification, reasons = _classify_row(
                    by_budget, verified_delta=delta, config=config
                )
                high_policies = tuple(
                    by_budget[budget]["policy"] for budget in config.budgets[-3:]
                )
                comparisons = {}
                for first, second in combinations(config.budgets, 2):
                    comparisons[f"{first}-{second}"] = _policy_comparison(
                        by_budget[first]["policy"], by_budget[second]["policy"]
                    )
                stored = _stored_policy(record)
                rows.append(
                    {
                        "partition": partition,
                        "game_id": record.game_id,
                        "game_index": record.game_index,
                        "ply": record.ply,
                        "raw_action": raw_action.uci,
                        "stored_action": _argmax(stored).uci,
                        "classification": classification,
                        "classification_reasons": reasons,
                        "verified_raw_value": verified[(index, raw_action)],
                        "verified_high_budget_value": verified[(index, final_action)],
                        "verified_delta_vs_raw": delta,
                        "maximum_high_budget_tv": max(
                            _tv(first, second)
                            for first, second in combinations(high_policies, 2)
                        ),
                        "minimum_high_budget_visit_margin": min(
                            float(by_budget[budget]["normalized_visit_margin"])
                            for budget in config.budgets[-3:]
                        ),
                        "stored_vs_64": _policy_comparison(
                            stored, by_budget[config.budgets[0]]["policy"]
                        ),
                        "pairwise": comparisons,
                        "budgets": {
                            str(budget): {
                                **{
                                    key: value.uci if isinstance(value, ChessMove) else value
                                    for key, value in by_budget[budget].items()
                                    if key != "policy"
                                },
                                "policy": tuple(
                                    (move.uci, probability)
                                    for move, probability in by_budget[budget]["policy"]
                                ),
                            }
                            for budget in config.budgets
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
    output = config.output_dir / "consistency.json"
    _atomic_json(
        output,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "learner_ablation_authorized": gate["passed"],
                "arena_authorized": False,
                "generation_authorized": False,
            },
            "rows": output_rows,
            "inference": asdict(batcher.statistics),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    path = run_teacher_consistency(
        TeacherConsistencyConfig(
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
