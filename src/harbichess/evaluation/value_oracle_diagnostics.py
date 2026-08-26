"""Frozen neural-versus-oracle leaf-value counterfactual for search teachers."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    _interval,
    _network_config,
    _source_commit,
    select_stratified_records,
)
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator, SearchEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.value_oracle import (
    DeterministicTacticalOracle,
    OracleValueEvaluator,
    TacticalOracleConfig,
)


@dataclass(frozen=True, slots=True)
class ValueOracleDiagnosticConfig:
    run_result: Path
    shard: Path
    output_dir: Path
    budgets: tuple[int, ...] = (64, 128, 256, 512, 800)
    positions: int = 32
    workers: int = 32
    seed: int = 2026082621
    oracle_depth: int = 2
    verifier_depth: int = 4
    material_scale: float = 39.0
    bootstrap_samples: int = 2_000

    def __post_init__(self) -> None:
        if (
            not self.budgets
            or tuple(sorted(set(self.budgets))) != self.budgets
            or any(budget <= 0 for budget in self.budgets)
            or min(
                self.positions,
                self.workers,
                self.oracle_depth,
                self.verifier_depth,
                self.bootstrap_samples,
            )
            <= 0
            or self.verifier_depth <= self.oracle_depth
            or self.material_scale <= 0
        ):
            raise ValueError("value oracle diagnostic configuration is invalid")


def _argmax_prior(evaluation) -> ChessMove:
    return min(evaluation.priors, key=lambda item: (-item[1], item[0].uci))[0]


def _search_rows(
    evaluator: SearchEvaluator,
    *,
    rules: PythonChessRules,
    records: tuple,
    budget: int,
    workers: int,
    seed: int,
    arm: str,
) -> tuple[dict[str, Any], ...]:
    search = MCTS(
        evaluator,
        rules=rules,
        config=SearchConfig(simulations=budget, dirichlet_fraction=0.0),
    )

    def inspect(index: int) -> dict[str, Any]:
        result = search.search(
            records[index].state,
            rng=random.Random(f"{seed}:{arm}:{budget}:{index}"),
            add_root_noise=False,
        )
        leader, runner = result.moves[:2]
        return {
            "selected_move": leader.move,
            "leader_visits": leader.visits,
            "leader_visit_share": leader.visits / budget,
            "visit_margin": leader.visits - runner.visits,
            "visited_children": sum(move.visits > 0 for move in result.moves),
            "legal_children": len(result.moves),
            "root_value": result.root_value,
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
        return tuple(pool.map(inspect, range(len(records))))


def _summarize_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    raw_moves: tuple[ChessMove, ...],
) -> dict[str, Any]:
    return {
        "raw_argmax_agreement": mean(
            row["selected_move"] == raw for row, raw in zip(rows, raw_moves, strict=True)
        ),
        "mean_leader_visits": mean(row["leader_visits"] for row in rows),
        "mean_leader_visit_share": mean(row["leader_visit_share"] for row in rows),
        "mean_visit_margin": mean(row["visit_margin"] for row in rows),
        "mean_visited_children": mean(row["visited_children"] for row in rows),
        "mean_legal_children": mean(row["legal_children"] for row in rows),
        "mean_absolute_root_value": mean(abs(row["root_value"]) for row in rows),
    }


def run_value_oracle_diagnostics(config: ValueOracleDiagnosticConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("value oracle diagnostics require a persisted baseline")
    rules = PythonChessRules()
    shard = read_shard(config.shard, rules=rules)
    records = select_stratified_records(
        shard.records,
        rules=rules,
        count=config.positions,
        seed=config.seed,
    )
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(baseline["path"]))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
        max_wait_seconds=0.00025,
    )
    neural = NeuralPositionEvaluator(batcher, rules=rules)
    shallow_oracle = DeterministicTacticalOracle(
        rules=rules,
        config=TacticalOracleConfig(
            depth=config.oracle_depth,
            material_scale=config.material_scale,
        ),
    )
    oracle = OracleValueEvaluator(neural, shallow_oracle)
    verifier = DeterministicTacticalOracle(
        rules=rules,
        config=TacticalOracleConfig(
            depth=config.verifier_depth,
            material_scale=config.material_scale,
        ),
    )
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
            raw = tuple(pool.map(lambda record: neural.evaluate(record.state), records))
        raw_moves = tuple(_argmax_prior(evaluation) for evaluation in raw)
        rows_by_arm: dict[str, dict[int, tuple[dict[str, Any], ...]]] = {
            "neural": {},
            "oracle": {},
        }
        for arm, evaluator in (("neural", neural), ("oracle", oracle)):
            for budget in config.budgets:
                rows_by_arm[arm][budget] = _search_rows(
                    evaluator,
                    rules=rules,
                    records=records,
                    budget=budget,
                    workers=config.workers,
                    seed=config.seed,
                    arm=arm,
                )

        actions = {
            (index, move)
            for index, move in enumerate(raw_moves)
        }
        actions.update(
            (index, row["selected_move"])
            for arm in rows_by_arm.values()
            for rows in arm.values()
            for index, row in enumerate(rows)
        )

        def verify(item: tuple[int, ChessMove]) -> tuple[tuple[int, ChessMove], float]:
            index, move = item
            child = rules.apply(records[index].state, move)
            return item, -verifier.value(child)

        with ThreadPoolExecutor(max_workers=min(config.workers, len(actions))) as pool:
            verified = dict(
                pool.map(verify, sorted(actions, key=lambda item: (item[0], item[1].uci)))
            )
        raw_verified = tuple(
            verified[(index, move)] for index, move in enumerate(raw_moves)
        )
        summaries: dict[str, dict[str, Any]] = {"neural": {}, "oracle": {}}
        for arm, budgets in rows_by_arm.items():
            for budget, rows in budgets.items():
                deltas = tuple(
                    verified[(index, row["selected_move"])] - raw_verified[index]
                    for index, row in enumerate(rows)
                )
                interval = _interval(
                    deltas,
                    samples=config.bootstrap_samples,
                    seed=config.seed + budget + (0 if arm == "neural" else 1_000_000),
                )
                summaries[arm][str(budget)] = {
                    **_summarize_rows(rows, raw_moves=raw_moves),
                    "mean_verified_action_value_delta": mean(deltas),
                    "verified_action_value_delta_95_interval": interval,
                    "positive_delta_ratio": mean(delta > 0 for delta in deltas),
                    "negative_delta_ratio": mean(delta < 0 for delta in deltas),
                    "counterfactual_teacher_qualified": interval[0] > 0.0,
                }

        tactical = {
            arm: run_tactical_sweep(
                evaluator,
                rules=rules,
                budgets=config.budgets,
                workers=config.workers,
                seed=config.seed,
            )
            for arm, evaluator in (("neural", neural), ("oracle", oracle))
        }
    finally:
        batcher.close()
    statistics = batcher.statistics
    qualified_oracle = tuple(
        int(budget)
        for budget, summary in summaries["oracle"].items()
        if summary["counterfactual_teacher_qualified"]
    )
    qualified_neural = tuple(
        int(budget)
        for budget, summary in summaries["neural"].items()
        if summary["counterfactual_teacher_qualified"]
    )
    path = config.output_dir / "diagnostics.json"
    _atomic_json(
        path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "baseline": baseline,
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "shard": str(config.shard),
                "output_dir": str(config.output_dir),
            },
            "controls": {
                "same_model": True,
                "same_positions": True,
                "same_policy_priors": True,
                "same_search_budgets": True,
                "same_search_config": True,
                "only_leaf_value_changed": True,
            },
            "search": summaries,
            "tactical": tactical,
            "gate": {
                "oracle_supports_value_bottleneck": bool(qualified_oracle),
                "qualified_oracle_budgets": qualified_oracle,
                "qualified_neural_budgets": qualified_neural,
                "continuous_learner_authorized": False,
                "generation_authorized": False,
                "note": "counterfactual oracle evidence cannot authorize training",
            },
            "inference": {
                **asdict(statistics),
                "average_batch_size": statistics.average_batch_size,
                "average_queue_wait_ms": statistics.average_queue_wait_ms,
            },
            "timing": {"elapsed_seconds": time.perf_counter() - started},
        },
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="64,128,256,512,800")
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026082621)
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--verifier-depth", type=int, default=4)
    parser.add_argument("--material-scale", type=float, default=39.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = run_value_oracle_diagnostics(
        ValueOracleDiagnosticConfig(
            run_result=arguments.run_result,
            shard=arguments.shard,
            output_dir=arguments.output_dir,
            budgets=tuple(int(value) for value in arguments.budgets.split(",") if value),
            positions=arguments.positions,
            workers=arguments.workers,
            seed=arguments.seed,
            oracle_depth=arguments.oracle_depth,
            verifier_depth=arguments.verifier_depth,
            material_scale=arguments.material_scale,
            bootstrap_samples=arguments.bootstrap_samples,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
