"""Generate confidence-gated continuation targets from independent branch searches."""

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
from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.schema import (
    BranchValueEstimate,
    ContinuationEvidence,
    ReplayRecord,
)
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig, SearchResult


@dataclass(frozen=True, slots=True)
class BranchEvidenceConfig:
    run_result: Path
    shard_paths: tuple[Path, ...]
    output_dir: Path
    root_simulations: int = 128
    branch_searches: int = 8
    branch_simulations: int = 64
    maximum_nonrepeat_branches: int = 3
    confidence_level: float = 0.95
    minimum_confident_advantage: float = 0.01
    workers: int = 96
    seed: int = 2026082505

    def __post_init__(self) -> None:
        counts = (
            self.root_simulations,
            self.branch_searches,
            self.branch_simulations,
            self.maximum_nonrepeat_branches,
            self.workers,
        )
        if any(value <= 0 for value in counts) or self.branch_searches <= 1:
            raise ValueError("branch evidence search counts must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("branch evidence confidence level must be in (0, 1)")
        if not 0.0 <= self.minimum_confident_advantage <= 1.0:
            raise ValueError("minimum confident advantage must be in [0, 1]")
        if not self.shard_paths or self.seed < 0:
            raise ValueError("branch evidence requires shards and a non-negative seed")


@dataclass(frozen=True, slots=True)
class RootBranchEvidence:
    game_id: str
    source_run: str
    root_value: float
    repeat_value: float
    repeat_actions: tuple[int, ...]
    branches: tuple[BranchValueEstimate, ...]
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


def _stable_seed(seed: int, game_id: str, move: str, search_index: int) -> int:
    payload = f"{seed}:{game_id}:{move}:{search_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def confidence_estimate(
    *,
    action: int,
    move: str,
    values: tuple[float, ...],
    confidence_level: float,
    comparisons: int,
) -> BranchValueEstimate:
    if len(values) <= 1 or comparisons <= 0:
        raise ValueError("confidence estimate requires repeated samples and comparisons")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    alpha = 1.0 - confidence_level
    critical = statistics.NormalDist().inv_cdf(1.0 - alpha / (2.0 * comparisons))
    return BranchValueEstimate(
        action=action,
        move=move,
        samples=len(values),
        mean_value=max(-1.0, min(1.0, mean)),
        standard_error=standard_error,
        lower_confidence_bound=max(-1.0, mean - critical * standard_error),
        upper_confidence_bound=min(1.0, mean + critical * standard_error),
    )


def _candidate_moves(
    record: ReplayRecord,
    search: SearchResult,
    repeat_moves: frozenset[ChessMove],
    rules: PythonChessRules,
    maximum: int,
) -> tuple[ChessMove, ...]:
    board = rules.board(record.state)
    move_by_action = {
        move_to_action(board, board.parse_uci(item.move.uci)): item.move for item in search.moves
    }
    selected = move_by_action.get(record.selected_action)
    candidates = [item.move for item in search.moves if item.move not in repeat_moves][:maximum]
    if selected is not None and selected not in repeat_moves and selected not in candidates:
        if len(candidates) == maximum:
            candidates[-1] = selected
        else:
            candidates.append(selected)
    return tuple(dict.fromkeys(candidates))


def build_confidence_target(
    record: ReplayRecord,
    evidence: ContinuationEvidence,
) -> ReplayRecord | None:
    if not evidence.qualified_actions:
        return None
    by_action = {branch.action: branch for branch in evidence.branches}
    risks_by_action = {risk.action: risk for risk in evidence.repetition_risks}
    if evidence.method_version >= 3:
        weights = {
            action: risks_by_action[action].risk_adjusted_value_lower_bound
            - evidence.repeat_value
            for action in evidence.qualified_actions
        }
    else:
        weights = {
            action: (by_action[action].lower_confidence_bound - evidence.repeat_value)
            * (1.0 - risks_by_action[action].upper_confidence_bound)
            if action in risks_by_action
            else by_action[action].lower_confidence_bound - evidence.repeat_value
            for action in evidence.qualified_actions
        }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("qualified confidence target must have positive total surplus")
    policy = tuple(sorted((action, weight / total) for action, weight in weights.items()))
    selected = max(evidence.qualified_actions, key=lambda action: (weights[action], -action))
    return replace(
        record,
        policy=policy,
        selected_action=selected,
        repetition_redirected=True,
        continuation_evidence=evidence,
    )


def run_branch_evidence(config: BranchEvidenceConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"branch evidence output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("branch evidence requires a persisted champion baseline")
    model_path = Path(baseline["path"])
    model_hash = _sha256(model_path)
    if model_hash != baseline["model_sha256"]:
        raise ValueError("branch evidence champion checksum mismatch")
    rules = PythonChessRules()
    shards = tuple((path, read_shard(path, rules=rules)) for path in config.shard_paths)
    merged = merge_continuation_replay(shards, recency_decay=1.0)
    source_by_game = {
        record.game_id: shard.header.run_id for _, shard in shards for record in shard.records
    }
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(model_path))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
        max_wait_seconds=0.00025,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    root_search = MCTS(
        evaluator,
        rules=rules,
        config=SearchConfig(simulations=config.root_simulations, dirichlet_fraction=0.0),
    )
    branch_search = MCTS(
        evaluator,
        rules=rules,
        config=SearchConfig(simulations=config.branch_simulations, dirichlet_fraction=0.25),
    )
    started = time.perf_counter()

    def search_root(record: ReplayRecord):
        result = root_search.search(record.state, rng=random.Random(0), add_root_noise=False)
        repeats = rules.claimable_threefold_moves(
            record.state, tuple(item.move for item in result.moves)
        )
        if not repeats:
            raise ValueError(f"branch evidence root has no repeat move: {record.game_id}")
        candidates = _candidate_moves(
            record,
            result,
            repeats,
            rules,
            config.maximum_nonrepeat_branches,
        )
        if not candidates:
            raise ValueError(f"branch evidence root has no non-repeat branch: {record.game_id}")
        return result, repeats, candidates

    try:
        with ThreadPoolExecutor(max_workers=min(config.workers, len(merged.records))) as pool:
            roots = tuple(pool.map(search_root, merged.records))

        tasks = tuple(
            (record, move)
            for record, (_, _, candidates) in zip(merged.records, roots, strict=True)
            for move in candidates
        )

        def evaluate_branch(item: tuple[ReplayRecord, ChessMove]):
            record, move = item
            child = rules.apply(record.state, move)
            outcome = rules.outcome(child, claim_draw=True)
            values = []
            for search_index in range(config.branch_searches):
                if outcome is not None:
                    value = float(outcome.value_for(record.side_to_move))
                else:
                    result = branch_search.search(
                        child,
                        rng=random.Random(
                            _stable_seed(config.seed, record.game_id, move.uci, search_index)
                        ),
                        add_root_noise=True,
                    )
                    value = -result.root_value
                values.append(value)
            board = rules.board(record.state)
            action = move_to_action(board, board.parse_uci(move.uci))
            return record.game_id, confidence_estimate(
                action=action,
                move=move.uci,
                values=tuple(values),
                confidence_level=config.confidence_level,
                comparisons=len(
                    next(
                        candidates
                        for candidate_record, (_, _, candidates) in zip(
                            merged.records, roots, strict=True
                        )
                        if candidate_record.game_id == record.game_id
                    )
                ),
            )

        with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
            branch_results = tuple(pool.map(evaluate_branch, tasks))
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    branches_by_game: dict[str, list[BranchValueEstimate]] = {}
    for game_id, estimate in branch_results:
        branches_by_game.setdefault(game_id, []).append(estimate)

    accepted = []
    audits = []
    for record, (root, repeats, _) in zip(merged.records, roots, strict=True):
        board = rules.board(record.state)
        repeat_actions = tuple(
            sorted(move_to_action(board, board.parse_uci(move.uci)) for move in repeats)
        )
        branches = tuple(sorted(branches_by_game[record.game_id], key=lambda item: item.action))
        qualified = tuple(
            branch.action
            for branch in branches
            if branch.lower_confidence_bound > config.minimum_confident_advantage
        )
        evidence = ContinuationEvidence(
            method_version=1,
            confidence_level=config.confidence_level,
            branch_searches=config.branch_searches,
            simulations_per_search=config.branch_simulations,
            repeat_value=0.0,
            minimum_advantage=config.minimum_confident_advantage,
            repeat_actions=repeat_actions,
            branches=branches,
            qualified_actions=qualified,
            source_model_sha256=model_hash,
        )
        target = build_confidence_target(record, evidence)
        if target is not None:
            target.validate_rules(rules)
            accepted.append(target)
        audits.append(
            RootBranchEvidence(
                game_id=record.game_id,
                source_run=source_by_game[record.game_id],
                root_value=root.root_value,
                repeat_value=0.0,
                repeat_actions=repeat_actions,
                branches=branches,
                qualified_actions=qualified,
                accepted=target is not None,
            )
        )

    config.output_dir.mkdir(parents=True)
    shard_path = config.output_dir / "continuation-confidence-gated.jsonl.gz"
    header = None
    if accepted:
        header = write_shard_atomic(
            shard_path,
            accepted,
            ShardMetadata(
                run_id=config.output_dir.name,
                generation=max(shard.header.generation for _, shard in shards) + 1,
                source_checkpoint=baseline["checkpoint_id"],
                source_commit=_source_commit(),
                created_at=_now(),
                split=ReplaySplit.TRAIN,
            ),
        )
    statistics = batcher.statistics
    result_path = config.output_dir / "branch-evidence.json"
    _atomic_json(
        result_path,
        {
            "created_at": _now(),
            "source_commit": _source_commit(),
            "champion": baseline,
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "shard_paths": [str(path) for path in config.shard_paths],
                "output_dir": str(config.output_dir),
            },
            "summary": {
                "roots": len(audits),
                "accepted_roots": len(accepted),
                "rejected_roots": len(audits) - len(accepted),
                "evaluated_branches": sum(len(item.branches) for item in audits),
                "qualified_branches": sum(len(item.qualified_actions) for item in audits),
                "elapsed_seconds": elapsed,
            },
            "inference": {
                **asdict(statistics),
                "average_batch_size": statistics.average_batch_size,
                "average_queue_wait_ms": statistics.average_queue_wait_ms,
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
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root-simulations", type=int, default=128)
    parser.add_argument("--branch-searches", type=int, default=8)
    parser.add_argument("--branch-simulations", type=int, default=64)
    parser.add_argument("--maximum-nonrepeat-branches", type=int, default=3)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-confident-advantage", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026082505)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_branch_evidence(
        BranchEvidenceConfig(
            run_result=arguments.run_result,
            shard_paths=tuple(arguments.shard),
            output_dir=arguments.output_dir,
            root_simulations=arguments.root_simulations,
            branch_searches=arguments.branch_searches,
            branch_simulations=arguments.branch_simulations,
            maximum_nonrepeat_branches=arguments.maximum_nonrepeat_branches,
            confidence_level=arguments.confidence_level,
            minimum_confident_advantage=arguments.minimum_confident_advantage,
            workers=arguments.workers,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
