"""Build continuation targets from repetition probability and loop value together."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.branch_evidence import build_confidence_target
from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.repetition_risk import (
    _atomic_json,
    _network_config,
    _now,
    _sha256,
    _source_commit,
    wilson_upper_bound,
)
from harbichess.replay.schema import (
    BranchValueEstimate,
    ContinuationEvidence,
    RepetitionRiskEstimate,
    ReplayRecord,
)
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig


@dataclass(frozen=True, slots=True)
class ValueAwareRiskConfig:
    source_result: Path
    input_shard: Path
    original_shards: tuple[Path, ...]
    output_dir: Path
    horizon_plies: int = 3
    rollouts: int = 16
    simulations: int = 32
    confidence_level: float = 0.95
    minimum_advantaged_root_value: float = 0.05
    workers: int = 96
    seed: int = 2026082507

    def __post_init__(self) -> None:
        if self.horizon_plies not in (2, 3):
            raise ValueError("value-aware repetition horizon must be two or three plies")
        if self.rollouts <= 0 or self.simulations <= 0 or self.workers <= 0:
            raise ValueError("value-aware repetition compute counts must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("value-aware confidence level must be in (0, 1)")
        if not 0.0 <= self.minimum_advantaged_root_value <= 1.0 or self.seed < 0:
            raise ValueError("value-aware root threshold and seed are invalid")
        if not self.original_shards:
            raise ValueError("value-aware targets require original continuation replay")


@dataclass(frozen=True, slots=True)
class RootValueRiskAudit:
    game_id: str
    root_value: float
    target_mode: str
    previous_qualified_actions: tuple[int, ...]
    repetition_risks: tuple[RepetitionRiskEstimate, ...]
    qualified_actions: tuple[int, ...]


def _stable_seed(seed: int, game_id: str, action: int, rollout: int) -> int:
    payload = f"{seed}:{game_id}:{action}:{rollout}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def loop_value_lower_bound(
    values: tuple[float, ...],
    confidence_level: float,
    exact: tuple[bool, ...] = (),
) -> tuple[float | None, float | None]:
    """Return mean and conservative lower bound for sparse loop-state values."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("loop value confidence level must be in (0, 1)")
    if not values:
        return None, None
    if exact and len(exact) != len(values):
        raise ValueError("loop value exactness must match the sampled values")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("loop values must be finite and bounded")
    mean = statistics.fmean(values)
    if exact and all(exact):
        return mean, min(values)
    if len(values) == 1:
        return mean, -1.0
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = statistics.NormalDist().inv_cdf(confidence_level)
    return mean, max(-1.0, mean - critical * standard_error)


def value_aware_risk_estimate(
    *,
    branch: BranchValueEstimate,
    horizon_plies: int,
    rollouts: int,
    loop_values: tuple[float, ...],
    exact_loop_values: tuple[bool, ...] = (),
    confidence_level: float,
    repeat_value: float,
) -> RepetitionRiskEstimate:
    events = len(loop_values)
    if exact_loop_values and len(exact_loop_values) != len(loop_values):
        raise ValueError("exact loop flags must match loop values")
    mean_loop, lower_loop = loop_value_lower_bound(
        loop_values, confidence_level, exact_loop_values
    )
    loop_floor = repeat_value if lower_loop is None else lower_loop
    observed_risk = events / rollouts
    adjusted = (
        (1.0 - observed_risk) * branch.lower_confidence_bound
        + observed_risk * min(branch.lower_confidence_bound, loop_floor)
    )
    return RepetitionRiskEstimate(
        action=branch.action,
        horizon_plies=horizon_plies,
        rollouts=rollouts,
        repetition_events=events,
        estimated_risk=observed_risk,
        upper_confidence_bound=wilson_upper_bound(events, rollouts, confidence_level),
        loop_value_samples=events,
        exact_loop_value_samples=sum(exact_loop_values),
        mean_loop_value=mean_loop,
        lower_loop_value_bound=lower_loop,
        risk_adjusted_value_lower_bound=max(-1.0, min(1.0, adjusted)),
    )


def value_aware_evidence(
    evidence: ContinuationEvidence,
    risks: tuple[RepetitionRiskEstimate, ...],
    *,
    root_value: float,
    minimum_advantaged_root_value: float,
) -> ContinuationEvidence:
    if root_value <= minimum_advantaged_root_value:
        raise ValueError("defensive or equal roots must preserve their original target")
    risks_by_action = {risk.action: risk for risk in risks}
    if set(risks_by_action) != set(evidence.qualified_actions):
        raise ValueError("value-aware risks must cover previously qualified branches")
    qualified = tuple(
        action
        for action in evidence.qualified_actions
        if risks_by_action[action].risk_adjusted_value_lower_bound is not None
        and risks_by_action[action].risk_adjusted_value_lower_bound
        > evidence.repeat_value + evidence.minimum_advantage
    )
    return replace(
        evidence,
        method_version=3,
        qualified_actions=qualified,
        repetition_risks=tuple(sorted(risks, key=lambda item: item.action)),
        maximum_repetition_risk=None,
        evaluated_root_value=root_value,
        minimum_advantaged_root_value=minimum_advantaged_root_value,
    )


def run_value_aware_risk(config: ValueAwareRiskConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"value-aware output already exists: {config.output_dir}")
    source = json.loads(config.source_result.read_text(encoding="utf-8"))
    baseline = source.get("baseline")
    if baseline is None:
        raise ValueError("value-aware risk requires a persisted champion baseline")
    model_path = Path(baseline["path"])
    if _sha256(model_path) != baseline["model_sha256"]:
        raise ValueError("value-aware risk champion checksum mismatch")
    rules = PythonChessRules()
    shard = read_shard(config.input_shard, rules=rules)
    if shard.header.target_schema != 4:
        raise ValueError("value-aware risk requires the immutable v4 source shard")
    if any(record.continuation_evidence is None for record in shard.records):
        raise ValueError("value-aware source records require branch evidence")
    originals = tuple(
        (path, read_shard(path, rules=rules)) for path in config.original_shards
    )
    merged_originals = merge_continuation_replay(originals, recency_decay=1.0)
    original_by_game = {record.game_id: record for record in merged_originals.records}
    if any(record.game_id not in original_by_game for record in shard.records):
        raise ValueError("original continuation replay does not cover every v4 root")

    network = HarbiChessNetwork(_network_config(source))
    network.load_weights(str(model_path))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
        max_wait_seconds=0.00025,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    rollout_search = MCTS(
        evaluator,
        rules=rules,
        config=SearchConfig(
            simulations=config.simulations,
            dirichlet_fraction=0.25,
            claim_draw=False,
        ),
    )
    value_search = MCTS(
        evaluator,
        rules=rules,
        config=SearchConfig(
            simulations=config.simulations,
            dirichlet_fraction=0.0,
            claim_draw=False,
        ),
    )
    tasks = tuple(
        (record, branch)
        for record in shard.records
        for branch in record.continuation_evidence.branches
        if branch.action in record.continuation_evidence.qualified_actions
    )
    started = time.perf_counter()

    def evaluate_branch(item: tuple[ReplayRecord, BranchValueEstimate]):
        record, branch = item
        loop_values = []
        exact_loop_values = []
        for rollout_index in range(config.rollouts):
            rng = random.Random(
                _stable_seed(config.seed, record.game_id, branch.action, rollout_index)
            )
            state = rules.apply(record.state, ChessMove(branch.move))
            for depth in range(1, config.horizon_plies + 1):
                result = rollout_search.search(state, rng=rng, add_root_noise=True)
                if not result.moves:
                    break
                state = rules.apply(
                    state, result.select_move(temperature=1.0, rng=rng)
                )
                if depth < 2:
                    continue
                board = rules.inspect(state)
                claimable = board.can_claim_threefold_repetition()
                if board.is_repetition(2) or claimable:
                    if claimable:
                        loop_value = 0.0
                        exact_loop_value = True
                    else:
                        loop_result = value_search.search(
                            state, rng=random.Random(0), add_root_noise=False
                        )
                        current_side = rules.view(state).side_to_move
                        loop_value = (
                            loop_result.root_value
                            if current_side is record.side_to_move
                            else -loop_result.root_value
                        )
                        exact_loop_value = False
                    loop_values.append(loop_value)
                    exact_loop_values.append(exact_loop_value)
                    break
        evidence = record.continuation_evidence
        assert evidence is not None
        return record.game_id, value_aware_risk_estimate(
            branch=branch,
            horizon_plies=config.horizon_plies,
            rollouts=config.rollouts,
            loop_values=tuple(loop_values),
            exact_loop_values=tuple(exact_loop_values),
            confidence_level=config.confidence_level,
            repeat_value=evidence.repeat_value,
        )

    try:
        with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
            results = tuple(pool.map(evaluate_branch, tasks))
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    risks_by_game: dict[str, list[RepetitionRiskEstimate]] = {}
    for game_id, risk in results:
        risks_by_game.setdefault(game_id, []).append(risk)

    targets = []
    audits = []
    for record in shard.records:
        previous = record.continuation_evidence
        assert previous is not None
        risks = tuple(risks_by_game[record.game_id])
        if record.root_value <= config.minimum_advantaged_root_value:
            target = original_by_game[record.game_id]
            mode = "original_defense_preserved"
            qualified: tuple[int, ...] = ()
        else:
            evidence = value_aware_evidence(
                previous,
                risks,
                root_value=record.root_value,
                minimum_advantaged_root_value=config.minimum_advantaged_root_value,
            )
            target = build_confidence_target(record, evidence)
            if target is None:
                target = original_by_game[record.game_id]
                mode = "original_value_uncertain"
            else:
                mode = "value_aware_redirect"
            qualified = evidence.qualified_actions
        target.validate_rules(rules)
        targets.append(target)
        audits.append(
            RootValueRiskAudit(
                game_id=record.game_id,
                root_value=record.root_value,
                target_mode=mode,
                previous_qualified_actions=previous.qualified_actions,
                repetition_risks=tuple(sorted(risks, key=lambda item: item.action)),
                qualified_actions=qualified,
            )
        )

    config.output_dir.mkdir(parents=True)
    shard_path = config.output_dir / "continuation-value-aware-risk.jsonl.gz"
    header = write_shard_atomic(
        shard_path,
        targets,
        ShardMetadata(
            run_id=config.output_dir.name,
            generation=shard.header.generation + 1,
            source_checkpoint=baseline["checkpoint_id"],
            source_commit=_source_commit(),
            created_at=_now(),
            split=ReplaySplit.TRAIN,
        ),
    )
    batch_statistics = batcher.statistics
    result_path = config.output_dir / "value-aware-risk.json"
    _atomic_json(
        result_path,
        {
            "created_at": _now(),
            "source_commit": _source_commit(),
            "champion": baseline,
            "source_shard": str(config.input_shard),
            "original_shards": [str(path) for path in config.original_shards],
            "config": {
                **asdict(config),
                "source_result": str(config.source_result),
                "input_shard": str(config.input_shard),
                "original_shards": [str(path) for path in config.original_shards],
                "output_dir": str(config.output_dir),
            },
            "summary": {
                "roots": len(audits),
                "original_defense_preserved": sum(
                    item.target_mode == "original_defense_preserved" for item in audits
                ),
                "original_value_uncertain": sum(
                    item.target_mode == "original_value_uncertain" for item in audits
                ),
                "value_aware_redirects": sum(
                    item.target_mode == "value_aware_redirect" for item in audits
                ),
                "evaluated_branches": len(tasks),
                "repetition_events": sum(
                    risk.repetition_events for item in audits for risk in item.repetition_risks
                ),
                "elapsed_seconds": elapsed,
            },
            "inference": {
                **asdict(batch_statistics),
                "average_batch_size": batch_statistics.average_batch_size,
                "average_queue_wait_ms": batch_statistics.average_queue_wait_ms,
            },
            "output_shard": {"path": str(shard_path), "header": asdict(header)},
            "roots": [asdict(item) for item in audits],
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--input-shard", required=True, type=Path)
    parser.add_argument("--original-shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon-plies", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-advantaged-root-value", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026082507)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_value_aware_risk(
        ValueAwareRiskConfig(
            source_result=arguments.source_result,
            input_shard=arguments.input_shard,
            original_shards=tuple(arguments.original_shard),
            output_dir=arguments.output_dir,
            horizon_plies=arguments.horizon_plies,
            rollouts=arguments.rollouts,
            simulations=arguments.simulations,
            confidence_level=arguments.confidence_level,
            minimum_advantaged_root_value=arguments.minimum_advantaged_root_value,
            workers=arguments.workers,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
