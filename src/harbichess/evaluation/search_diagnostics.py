"""Audit champion search conventions and tactical budget scaling without training."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.evaluation.teacher_qualification import select_stratified_records
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import TACTICAL_CASES, run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator, PositionEvaluation
from harbichess.search.mcts import MCTS, SearchConfig, SearchNode


@dataclass(frozen=True, slots=True)
class SearchDiagnosticConfig:
    run_result: Path
    shard: Path
    output_dir: Path
    budgets: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256, 512, 800)
    replay_positions: int = 32
    workers: int = 32
    seed: int = 2026082621

    def __post_init__(self) -> None:
        if (
            not self.budgets
            or tuple(sorted(set(self.budgets))) != self.budgets
            or any(budget <= 0 for budget in self.budgets)
            or self.replay_positions <= 0
            or self.workers <= 0
        ):
            raise ValueError("search diagnostic configuration is invalid")


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
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


def _network_config(payload: Mapping[str, Any]) -> NetworkConfig:
    config = payload["config"]
    return NetworkConfig(
        trunk_channels=int(config["trunk_channels"]),
        residual_blocks=int(config["residual_blocks"]),
        policy_channels=int(config["policy_channels"]),
        value_channels=int(config["value_channels"]),
        value_hidden=int(config["value_hidden"]),
    )


def audit_search_conventions() -> dict[str, Any]:
    """Exercise sign, terminal, history, and FPU conventions independently of MLX."""

    rules = PythonChessRules()
    checkmated = rules.initial_state()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        checkmated = rules.apply(checkmated, ChessMove(move))
    outcome = rules.outcome(checkmated)
    assert outcome is not None
    terminal_value = outcome.value_for(rules.view(checkmated).side_to_move)

    path = [SearchNode(), SearchNode(), SearchNode(), SearchNode()]
    MCTS._backpropagate(path, -1.0)
    backed_up = tuple(node.value_sum for node in path)

    repeated = rules.initial_state()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeated = rules.apply(repeated, ChessMove(move))
    repeated_outcome = rules.outcome(repeated, claim_draw=True)
    fen_only = rules.initial_state(rules.view(repeated).fen)
    fen_only_outcome = rules.outcome(fen_only, claim_draw=True)

    checks = {
        "terminal_value_is_side_to_move_loss": terminal_value == -1,
        "backup_alternates_every_ply": backed_up == (1.0, -1.0, 1.0, -1.0),
        "history_state_claims_threefold": (
            repeated_outcome is not None
            and repeated_outcome.termination == "threefold_repetition"
        ),
        "fen_only_does_not_invent_threefold": fen_only_outcome is None,
        "unvisited_child_fpu_is_zero": SearchNode().mean_value == 0.0,
        "virtual_loss_is_not_used": True,
        "transposition_table_is_not_used": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "terminal_value": terminal_value,
        "backed_up_values_root_to_leaf": backed_up,
        "value_perspective": "side-to-move",
        "child_q_to_parent_q": "negate child mean value",
        "fpu": "unvisited child mean value is zero",
        "batching_model": "independent trees share inference batches; trees are not shared",
    }


def _argmax_prior(evaluation: PositionEvaluation) -> ChessMove:
    return min(evaluation.priors, key=lambda item: (-item[1], item[0].uci))[0]


def _evaluation_distance(
    first: PositionEvaluation,
    second: PositionEvaluation,
) -> tuple[float, float]:
    left = dict(first.priors)
    right = dict(second.priors)
    policy = max(abs(left[move] - right[move]) for move in left.keys() | right.keys())
    return policy, abs(first.value - second.value)


def audit_batching(
    evaluator: NeuralPositionEvaluator,
    *,
    rules: PythonChessRules,
    workers: int,
) -> dict[str, Any]:
    """Compare serial and coalesced inference outputs on identical encoded states."""

    states = tuple(rules.initial_state(case.fen) for case in TACTICAL_CASES)
    serial = tuple(evaluator.evaluate(state) for state in states)
    with ThreadPoolExecutor(max_workers=min(workers, len(states))) as pool:
        parallel = tuple(pool.map(evaluator.evaluate, states))
    distances = tuple(
        _evaluation_distance(left, right)
        for left, right in zip(serial, parallel, strict=True)
    )
    maximum_policy = max(distance[0] for distance in distances)
    maximum_value = max(distance[1] for distance in distances)
    return {
        "equivalent": maximum_policy <= 1e-7 and maximum_value <= 1e-7,
        "maximum_policy_probability_delta": maximum_policy,
        "maximum_value_delta": maximum_value,
        "positions": len(states),
    }


def profile_replay_search(
    evaluator: NeuralPositionEvaluator,
    *,
    rules: PythonChessRules,
    records: tuple[ReplayRecord, ...],
    budgets: tuple[int, ...],
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Measure value signal, branch coverage, and root-score separation on replay."""

    with ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
        raw = tuple(pool.map(lambda record: evaluator.evaluate(record.state), records))
    raw_moves = tuple(_argmax_prior(evaluation) for evaluation in raw)
    raw_values = tuple(evaluation.value for evaluation in raw)
    result: dict[str, Any] = {
        "positions": len(records),
        "mean_raw_value": mean(raw_values),
        "mean_absolute_raw_value": mean(abs(value) for value in raw_values),
        "raw_value_signs": {
            "positive": sum(value > 0 for value in raw_values),
            "zero": sum(value == 0 for value in raw_values),
            "negative": sum(value < 0 for value in raw_values),
        },
        "mean_top_policy_prior": mean(
            max(prior for _, prior in evaluation.priors) for evaluation in raw
        ),
        "budgets": [],
    }
    for budget in budgets:
        search = MCTS(
            evaluator,
            rules=rules,
            config=SearchConfig(simulations=budget, dirichlet_fraction=0.0),
        )

        def inspect(index: int, search: MCTS = search, budget: int = budget) -> dict[str, Any]:
            search_result = search.search(
                records[index].state,
                rng=random.Random(f"{seed}:replay:{index}:{budget}"),
                add_root_noise=False,
            )
            leader, runner = search_result.moves[:2]
            alternatives = tuple(move.mean_value for move in search_result.moves[1:])
            return {
                "raw_agreement": leader.move == raw_moves[index],
                "leader_visits": leader.visits,
                "leader_visit_share": leader.visits / budget,
                "visit_margin": leader.visits - runner.visits,
                "q_margin_over_visit_runner": leader.mean_value - runner.mean_value,
                "q_margin_over_best_alternative": leader.mean_value - max(alternatives),
                "leader_is_best_q": leader.mean_value >= max(alternatives),
                "absolute_leader_q": abs(leader.mean_value),
                "visited_children": sum(move.visits > 0 for move in search_result.moves),
                "legal_children": len(search_result.moves),
            }

        with ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
            rows = tuple(pool.map(inspect, range(len(records))))
        result["budgets"].append(
            {
                "budget": budget,
                "raw_argmax_agreement": mean(row["raw_agreement"] for row in rows),
                "mean_leader_visits": mean(row["leader_visits"] for row in rows),
                "mean_leader_visit_share": mean(row["leader_visit_share"] for row in rows),
                "mean_visit_margin": mean(row["visit_margin"] for row in rows),
                "mean_q_margin_over_visit_runner": mean(
                    row["q_margin_over_visit_runner"] for row in rows
                ),
                "mean_q_margin_over_best_alternative": mean(
                    row["q_margin_over_best_alternative"] for row in rows
                ),
                "leader_is_best_q_ratio": mean(row["leader_is_best_q"] for row in rows),
                "mean_absolute_leader_q": mean(row["absolute_leader_q"] for row in rows),
                "mean_visited_children": mean(row["visited_children"] for row in rows),
                "mean_legal_children": mean(row["legal_children"] for row in rows),
            }
        )
    return result


def run_search_diagnostics(config: SearchDiagnosticConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("search diagnostics require a persisted baseline model")
    rules = PythonChessRules()
    shard = read_shard(config.shard, rules=rules)
    records = select_stratified_records(
        shard.records,
        rules=rules,
        count=config.replay_positions,
        seed=config.seed,
    )
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(baseline["path"]))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
        max_wait_seconds=0.00025,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    started = time.perf_counter()
    try:
        conventions = audit_search_conventions()
        batching = audit_batching(evaluator, rules=rules, workers=config.workers)
        tactical = run_tactical_sweep(
            evaluator,
            rules=rules,
            budgets=config.budgets,
            workers=config.workers,
            seed=config.seed,
        )
        replay = profile_replay_search(
            evaluator,
            rules=rules,
            records=records,
            budgets=tuple(budget for budget in config.budgets if budget >= 64),
            workers=config.workers,
            seed=config.seed,
        )
    finally:
        batcher.close()
    statistics = batcher.statistics
    elapsed = time.perf_counter() - started
    path = config.output_dir / "diagnostics.json"
    _atomic_json(
        path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "run_result": str(config.run_result),
            "shard": str(config.shard),
            "baseline": baseline,
            "config": {
                "run_result": str(config.run_result),
                "shard": str(config.shard),
                "output_dir": str(config.output_dir),
                "budgets": config.budgets,
                "replay_positions": config.replay_positions,
                "workers": config.workers,
                "seed": config.seed,
            },
            "conventions": conventions,
            "batching": batching,
            "tactical": tactical,
            "replay_search": replay,
            "inference": {
                **asdict(statistics),
                "average_batch_size": statistics.average_batch_size,
                "average_queue_wait_ms": statistics.average_queue_wait_ms,
            },
            "timing": {"elapsed_seconds": elapsed},
            "gate": {
                "continuous_learner_authorized": False,
                "generation_authorized": False,
                "note": "diagnostics never authorize training or generation",
            },
        },
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="1,4,8,16,32,64,128,256,512,800")
    parser.add_argument("--replay-positions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026082621)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = run_search_diagnostics(
        SearchDiagnosticConfig(
            run_result=arguments.run_result,
            shard=arguments.shard,
            output_dir=arguments.output_dir,
            budgets=tuple(int(value) for value in arguments.budgets.split(",") if value),
            replay_positions=arguments.replay_positions,
            workers=arguments.workers,
            seed=arguments.seed,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
