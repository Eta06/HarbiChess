"""Gate v4 continuation branches with short-horizon repetition rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.branch_evidence import build_confidence_target
from harbichess.replay.schema import (
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
class RepetitionRiskConfig:
    source_result: Path
    input_shard: Path
    output_dir: Path
    horizon_plies: int = 3
    rollouts: int = 16
    simulations: int = 32
    confidence_level: float = 0.95
    maximum_repetition_risk: float = 0.20
    workers: int = 96
    seed: int = 2026082507

    def __post_init__(self) -> None:
        if self.horizon_plies not in (2, 3):
            raise ValueError("repetition risk horizon must be two or three plies")
        if self.rollouts <= 0 or self.simulations <= 0 or self.workers <= 0:
            raise ValueError("repetition risk compute counts must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("repetition risk confidence level must be in (0, 1)")
        if not 0.0 <= self.maximum_repetition_risk <= 1.0 or self.seed < 0:
            raise ValueError("repetition risk threshold and seed are invalid")


@dataclass(frozen=True, slots=True)
class RootRiskAudit:
    game_id: str
    previous_qualified_actions: tuple[int, ...]
    repetition_risks: tuple[RepetitionRiskEstimate, ...]
    qualified_actions: tuple[int, ...]
    accepted: bool


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    config = payload["config"]
    return NetworkConfig(
        trunk_channels=int(config["trunk_channels"]),
        residual_blocks=int(config["residual_blocks"]),
        policy_channels=int(config["policy_channels"]),
        value_channels=int(config["value_channels"]),
        value_hidden=int(config["value_hidden"]),
    )


def _stable_seed(seed: int, game_id: str, action: int, rollout: int) -> int:
    payload = f"{seed}:{game_id}:{action}:{rollout}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def wilson_upper_bound(events: int, trials: int, confidence_level: float) -> float:
    """One-sided Wilson upper confidence bound for a Bernoulli risk."""

    if not 0 <= events <= trials or trials <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Wilson bound inputs are invalid")
    probability = events / trials
    critical = statistics.NormalDist().inv_cdf(confidence_level)
    squared = critical * critical
    denominator = 1.0 + squared / trials
    center = (probability + squared / (2.0 * trials)) / denominator
    half_width = critical * math.sqrt(
        probability * (1.0 - probability) / trials + squared / (4.0 * trials * trials)
    ) / denominator
    return min(1.0, center + half_width)


def risk_estimate(
    *,
    action: int,
    horizon_plies: int,
    events: int,
    rollouts: int,
    confidence_level: float,
) -> RepetitionRiskEstimate:
    return RepetitionRiskEstimate(
        action=action,
        horizon_plies=horizon_plies,
        rollouts=rollouts,
        repetition_events=events,
        estimated_risk=events / rollouts,
        upper_confidence_bound=wilson_upper_bound(events, rollouts, confidence_level),
    )


def risk_gated_evidence(
    evidence: ContinuationEvidence,
    risks: tuple[RepetitionRiskEstimate, ...],
    *,
    maximum_repetition_risk: float,
) -> ContinuationEvidence:
    risks_by_action = {risk.action: risk for risk in risks}
    if set(risks_by_action) != set(evidence.qualified_actions):
        raise ValueError("risk estimates must exactly cover previously qualified branches")
    qualified = tuple(
        action
        for action in evidence.qualified_actions
        if risks_by_action[action].upper_confidence_bound <= maximum_repetition_risk
    )
    return replace(
        evidence,
        method_version=2,
        qualified_actions=qualified,
        repetition_risks=tuple(sorted(risks, key=lambda item: item.action)),
        maximum_repetition_risk=maximum_repetition_risk,
    )


def run_repetition_risk(config: RepetitionRiskConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"repetition risk output already exists: {config.output_dir}")
    source = json.loads(config.source_result.read_text(encoding="utf-8"))
    baseline = source.get("baseline")
    if baseline is None:
        raise ValueError("repetition risk requires a persisted champion baseline")
    model_path = Path(baseline["path"])
    if _sha256(model_path) != baseline["model_sha256"]:
        raise ValueError("repetition risk champion checksum mismatch")
    rules = PythonChessRules()
    shard = read_shard(config.input_shard, rules=rules)
    if shard.header.target_schema != 4:
        raise ValueError("repetition risk requires an immutable v4 source shard")
    if any(record.continuation_evidence is None for record in shard.records):
        raise ValueError("repetition risk source records require branch evidence")

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
    tasks = tuple(
        (record, branch.action, branch.move)
        for record in shard.records
        for branch in record.continuation_evidence.branches
        if branch.action in record.continuation_evidence.qualified_actions
    )
    started = time.perf_counter()

    def evaluate_branch(item: tuple[ReplayRecord, int, str]):
        record, action, move = item
        events = 0
        for rollout_index in range(config.rollouts):
            rng = random.Random(_stable_seed(config.seed, record.game_id, action, rollout_index))
            state = rules.apply(record.state, ChessMove(move))
            repeated = False
            for depth in range(1, config.horizon_plies + 1):
                result = rollout_search.search(state, rng=rng, add_root_noise=True)
                if not result.moves:
                    break
                selected = result.select_move(temperature=1.0, rng=rng)
                state = rules.apply(state, selected)
                if depth >= 2:
                    board = rules.inspect(state)
                    if board.is_repetition(2) or board.can_claim_threefold_repetition():
                        repeated = True
                        break
            events += int(repeated)
        return record.game_id, risk_estimate(
            action=action,
            horizon_plies=config.horizon_plies,
            events=events,
            rollouts=config.rollouts,
            confidence_level=config.confidence_level,
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

    accepted = []
    audits = []
    for record in shard.records:
        previous = record.continuation_evidence
        assert previous is not None
        risks = tuple(risks_by_game[record.game_id])
        evidence = risk_gated_evidence(
            previous,
            risks,
            maximum_repetition_risk=config.maximum_repetition_risk,
        )
        target = build_confidence_target(record, evidence)
        if target is not None:
            target.validate_rules(rules)
            accepted.append(target)
        audits.append(
            RootRiskAudit(
                game_id=record.game_id,
                previous_qualified_actions=previous.qualified_actions,
                repetition_risks=evidence.repetition_risks,
                qualified_actions=evidence.qualified_actions,
                accepted=target is not None,
            )
        )

    config.output_dir.mkdir(parents=True)
    shard_path = config.output_dir / "continuation-repetition-risk-gated.jsonl.gz"
    header = None
    if accepted:
        header = write_shard_atomic(
            shard_path,
            accepted,
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
    result_path = config.output_dir / "repetition-risk.json"
    _atomic_json(
        result_path,
        {
            "created_at": _now(),
            "source_commit": _source_commit(),
            "champion": baseline,
            "source_shard": str(config.input_shard),
            "config": {
                **asdict(config),
                "source_result": str(config.source_result),
                "input_shard": str(config.input_shard),
                "output_dir": str(config.output_dir),
            },
            "summary": {
                "roots": len(audits),
                "accepted_roots": len(accepted),
                "rejected_roots": len(audits) - len(accepted),
                "evaluated_branches": len(tasks),
                "accepted_branches": sum(len(item.qualified_actions) for item in audits),
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
            "output_shard": (
                {"path": str(shard_path), "header": asdict(header)} if header else None
            ),
            "roots": [asdict(item) for item in audits],
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--input-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon-plies", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--maximum-repetition-risk", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026082507)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_repetition_risk(
        RepetitionRiskConfig(
            source_result=arguments.source_result,
            input_shard=arguments.input_shard,
            output_dir=arguments.output_dir,
            horizon_plies=arguments.horizon_plies,
            rollouts=arguments.rollouts,
            simulations=arguments.simulations,
            confidence_level=arguments.confidence_level,
            maximum_repetition_risk=arguments.maximum_repetition_risk,
            workers=arguments.workers,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
