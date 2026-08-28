"""Audit whether noisy stored replay targets match the qualified clean search teacher."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
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
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.targets import prune_noise_attributable_visits, visit_policy
from harbichess.search.value_oracle import (
    DeterministicTacticalOracle,
    OracleValueEvaluator,
    TacticalOracleConfig,
)

Policy = tuple[tuple[ChessMove, float], ...]


@dataclass(frozen=True, slots=True)
class ReplayTeacherAlignmentConfig:
    run_result: Path
    shard: Path
    output_dir: Path
    positions: int = 48
    simulations: int = 64
    workers: int = 32
    seed: int = 2026082804
    oracle_depth: int = 1
    verifier_depth: int = 4
    bootstrap_samples: int = 2_000

    def __post_init__(self) -> None:
        if min(
            self.positions,
            self.simulations,
            self.workers,
            self.oracle_depth,
            self.verifier_depth,
            self.bootstrap_samples,
        ) <= 0 or self.verifier_depth <= self.oracle_depth:
            raise ValueError("replay teacher alignment configuration is invalid")


def _policy(record: ReplayRecord, items: tuple[tuple[int, float], ...]) -> Policy:
    board = PythonChessRules().board(record.state)
    return tuple(
        (ChessMove(action_to_legal_move(board, action).uci()), value)
        for action, value in items
    )


def _argmax(policy: Policy) -> ChessMove:
    return min(policy, key=lambda item: (-item[1], item[0].uci))[0]


def _tv(first: Policy, second: Policy) -> float:
    left, right = dict(first), dict(second)
    return 0.5 * sum(
        abs(left.get(move, 0.0) - right.get(move, 0.0))
        for move in left.keys() | right.keys()
    )


def _kl(target: Policy, reference: Policy) -> float:
    probabilities = dict(reference)
    return sum(
        probability * math.log(probability / max(probabilities.get(move, 0.0), 1e-12))
        for move, probability in target
        if probability > 0
    )


def run_replay_teacher_alignment(config: ReplayTeacherAlignmentConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"alignment output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not run.get("passed") or run.get("mode") != "generation_only":
        raise ValueError("alignment requires a qualified generation-only replay")
    rules = PythonChessRules()
    records = select_stratified_records(
        read_shard(config.shard, rules=rules).records,
        rules=rules,
        count=config.positions,
        seed=config.seed,
    )
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(run["baseline"]["path"]))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
    )
    neural = NeuralPositionEvaluator(batcher, rules=rules)
    teacher = OracleValueEvaluator(
        neural,
        DeterministicTacticalOracle(
            rules=rules,
            config=TacticalOracleConfig(depth=config.oracle_depth),
        ),
    )
    clean_search = MCTS(
        teacher,
        rules=rules,
        config=SearchConfig(simulations=config.simulations, dirichlet_fraction=0.0),
    )
    verifier = DeterministicTacticalOracle(
        rules=rules,
        config=TacticalOracleConfig(depth=config.verifier_depth),
    )
    started = time.perf_counter()

    def inspect(item: tuple[int, ReplayRecord]) -> dict[str, object]:
        index, record = item
        stored = _policy(record, record.policy)
        raw = _policy(record, record.raw_policy)
        clean = visit_policy(
            clean_search.search(
                record.state,
                rng=random.Random(f"{config.seed}:clean:{index}"),
                add_root_noise=False,
            )
        )
        noisy_result = clean_search.search(
            record.state,
            rng=random.Random(f"{config.seed}:noisy:{index}"),
            add_root_noise=True,
        )
        noisy = visit_policy(noisy_result)
        pruned = prune_noise_attributable_visits(
            noisy_result,
            dict(noisy_result.network_priors),
        )
        moves = {
            "raw": _argmax(raw),
            "stored": _argmax(stored),
            "noisy": _argmax(noisy),
            "pruned": _argmax(pruned),
            "clean": _argmax(clean),
        }
        verified = {
            name: -verifier.value(rules.apply(record.state, move))
            for name, move in moves.items()
        }
        return {
            "game_id": record.game_id,
            "ply": record.ply,
            "raw_action": moves["raw"].uci,
            "stored_action": moves["stored"].uci,
            "noisy_action": moves["noisy"].uci,
            "pruned_action": moves["pruned"].uci,
            "clean_action": moves["clean"].uci,
            "stored_clean_agree": moves["stored"] == moves["clean"],
            "raw_clean_agree": moves["raw"] == moves["clean"],
            "stored_raw_agree": moves["stored"] == moves["raw"],
            "noisy_clean_agree": moves["noisy"] == moves["clean"],
            "pruned_clean_agree": moves["pruned"] == moves["clean"],
            "stored_clean_tv": _tv(stored, clean),
            "noisy_clean_tv": _tv(noisy, clean),
            "pruned_clean_tv": _tv(pruned, clean),
            "stored_clean_kl": _kl(stored, clean),
            "clean_stored_kl": _kl(clean, stored),
            "verified": verified,
            "stored_delta_vs_raw": verified["stored"] - verified["raw"],
            "noisy_delta_vs_raw": verified["noisy"] - verified["raw"],
            "pruned_delta_vs_raw": verified["pruned"] - verified["raw"],
            "clean_delta_vs_raw": verified["clean"] - verified["raw"],
            "stored_delta_vs_clean": verified["stored"] - verified["clean"],
        }

    try:
        with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
            rows = tuple(pool.map(inspect, enumerate(records)))
    finally:
        batcher.close()

    def values(name: str) -> tuple[float, ...]:
        return tuple(float(row[name]) for row in rows)

    stored_deltas = values("stored_delta_vs_raw")
    noisy_deltas = values("noisy_delta_vs_raw")
    pruned_deltas = values("pruned_delta_vs_raw")
    clean_deltas = values("clean_delta_vs_raw")
    stored_vs_clean = values("stored_delta_vs_clean")
    output = config.output_dir / "alignment.json"
    _atomic_json(
        output,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "shard": str(config.shard),
                "output_dir": str(config.output_dir),
            },
            "summary": {
                "positions": len(rows),
                "stored_clean_top_action_agreement": mean(
                    bool(row["stored_clean_agree"]) for row in rows
                ),
                "raw_clean_top_action_agreement": mean(
                    bool(row["raw_clean_agree"]) for row in rows
                ),
                "stored_raw_top_action_agreement": mean(
                    bool(row["stored_raw_agree"]) for row in rows
                ),
                "noisy_clean_top_action_agreement": mean(
                    bool(row["noisy_clean_agree"]) for row in rows
                ),
                "pruned_clean_top_action_agreement": mean(
                    bool(row["pruned_clean_agree"]) for row in rows
                ),
                "mean_stored_clean_tv": mean(values("stored_clean_tv")),
                "mean_noisy_clean_tv": mean(values("noisy_clean_tv")),
                "mean_pruned_clean_tv": mean(values("pruned_clean_tv")),
                "mean_stored_clean_kl": mean(values("stored_clean_kl")),
                "mean_clean_stored_kl": mean(values("clean_stored_kl")),
                "stored_verified_delta_vs_raw": mean(stored_deltas),
                "stored_verified_delta_vs_raw_95_interval": _interval(
                    stored_deltas,
                    samples=config.bootstrap_samples,
                    seed=config.seed,
                ),
                "clean_verified_delta_vs_raw": mean(clean_deltas),
                "clean_verified_delta_vs_raw_95_interval": _interval(
                    clean_deltas,
                    samples=config.bootstrap_samples,
                    seed=config.seed + 1,
                ),
                "stored_verified_delta_vs_clean": mean(stored_vs_clean),
                "stored_verified_delta_vs_clean_95_interval": _interval(
                    stored_vs_clean,
                    samples=config.bootstrap_samples,
                    seed=config.seed + 2,
                ),
                "noisy_verified_delta_vs_raw": mean(noisy_deltas),
                "noisy_verified_delta_vs_raw_95_interval": _interval(
                    noisy_deltas,
                    samples=config.bootstrap_samples,
                    seed=config.seed + 3,
                ),
                "pruned_verified_delta_vs_raw": mean(pruned_deltas),
                "pruned_verified_delta_vs_raw_95_interval": _interval(
                    pruned_deltas,
                    samples=config.bootstrap_samples,
                    seed=config.seed + 4,
                ),
            },
            "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = run_replay_teacher_alignment(
        ReplayTeacherAlignmentConfig(
            run_result=arguments.run_result,
            shard=arguments.shard,
            output_dir=arguments.output_dir,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
