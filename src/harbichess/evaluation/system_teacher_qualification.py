"""System-level qualification of clean search against the raw network policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessState, GameOutcome, Side, TerminalResult
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.arena import _openings
from harbichess.search.batching import BatchStatistics, SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator, SearchEvaluator
from harbichess.search.mcts import MCTS, MoveStatistics, SearchConfig, SearchResult


@dataclass(frozen=True, slots=True)
class SystemTeacherConfig:
    output_dir: Path
    model_path: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    model_sha256: str = "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    trunk_channels: int = 16
    residual_blocks: int = 2
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden: int = 32
    budgets: tuple[int, ...] = (64, 128, 256)
    opening_pairs: int = 32
    opening_plies: int = 8
    max_plies: int = 256
    workers: int = 24
    seed: int = 2026082867
    inference_wait_seconds: float = 0.00025
    bootstrap_samples: int = 50_000

    def __post_init__(self) -> None:
        counts = (
            *self.budgets,
            self.opening_pairs,
            self.max_plies,
            self.workers,
            self.bootstrap_samples,
        )
        if any(value <= 0 for value in counts) or self.opening_plies < 0:
            raise ValueError("system teacher counts must be positive")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("system teacher budgets must be unique and increasing")
        if self.seed < 0 or self.inference_wait_seconds < 0:
            raise ValueError("system teacher seed and wait must be non-negative")


@dataclass(frozen=True, slots=True)
class QualificationGame:
    pair_index: int
    candidate_side: Side | None
    opening_moves: tuple[str, ...]
    outcome: GameOutcome
    candidate_score: float | None
    plies: int


class RawPolicy:
    """Search-compatible deterministic legal argmax from one network evaluation."""

    def __init__(self, evaluator: SearchEvaluator) -> None:
        self.evaluator = evaluator

    def search(
        self,
        state: ChessState,
        *,
        rng: random.Random,
        add_root_noise: bool = False,
    ) -> SearchResult:
        del rng
        if add_root_noise:
            raise ValueError("raw policy does not support root noise")
        evaluation = self.evaluator.evaluate(state)
        leader = min(evaluation.priors, key=lambda item: (-item[1], item[0].uci))[0]
        moves = tuple(
            MoveStatistics(move, int(move == leader), prior, evaluation.value)
            for move, prior in evaluation.priors
        )
        return SearchResult(
            moves=tuple(sorted(moves, key=lambda item: (-item.visits, item.move.uci))),
            root_value=evaluation.value,
            simulations=0,
            network_priors=evaluation.priors,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _score(outcome: GameOutcome, side: Side) -> float:
    value = outcome.value_for(side)
    return 1.0 if value > 0 else 0.0 if value < 0 else 0.5


def _play_game(
    white: RawPolicy | MCTS,
    black: RawPolicy | MCTS,
    rules: PythonChessRules,
    initial_state: ChessState,
    *,
    pair_index: int,
    candidate_side: Side | None,
    opening_moves: tuple[str, ...],
    max_plies: int,
) -> QualificationGame:
    state = initial_state
    while True:
        outcome = rules.outcome(state, claim_draw=True)
        if outcome is not None:
            break
        if state.ply >= max_plies:
            outcome = GameOutcome(TerminalResult.DRAW, "max_plies")
            break
        player = white if rules.view(state).side_to_move is Side.WHITE else black
        search = player.search(state, rng=random.Random(0), add_root_noise=False)
        move = search.select_move(temperature=0.0, rng=random.Random(0))
        state = rules.apply(state, move)
    return QualificationGame(
        pair_index=pair_index,
        candidate_side=candidate_side,
        opening_moves=opening_moves,
        outcome=outcome,
        candidate_score=None if candidate_side is None else _score(outcome, candidate_side),
        plies=state.ply,
    )


def _bootstrap_interval(
    pair_scores: tuple[float, ...], *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not pair_scores:
        raise ValueError("bootstrap requires paired scores")
    estimate = sum(pair_scores) / len(pair_scores)
    rng = random.Random(seed)
    draws = sorted(
        sum(pair_scores[rng.randrange(len(pair_scores))] for _ in pair_scores)
        / len(pair_scores)
        for _ in range(samples)
    )
    return estimate, draws[round(0.025 * (samples - 1))], draws[round(0.975 * (samples - 1))]


def summarize_games(
    games: tuple[QualificationGame, ...], *, bootstrap_samples: int, seed: int
) -> dict[str, object]:
    scores = [game.candidate_score for game in games]
    if any(score is None for score in scores):
        raise ValueError("candidate summary requires candidate games")
    numeric = [float(score) for score in scores if score is not None]
    pairs: dict[int, list[float]] = {}
    for game, score in zip(games, numeric, strict=True):
        pairs.setdefault(game.pair_index, []).append(score)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("every opening must have exactly two color-balanced games")
    pair_scores = tuple(sum(pair) / 2.0 for _, pair in sorted(pairs.items()))
    estimate, low, high = _bootstrap_interval(
        pair_scores, samples=bootstrap_samples, seed=seed
    )
    decisive = [score for score in numeric if score != 0.5]
    return {
        "games": len(games),
        "wins": sum(score == 1.0 for score in numeric),
        "draws": sum(score == 0.5 for score in numeric),
        "losses": sum(score == 0.0 for score in numeric),
        "score_rate": estimate,
        "score_interval": {"low": low, "high": high},
        "decisive_score": sum(decisive) / len(decisive) if decisive else 0.5,
        "decisive_games": len(decisive),
        "max_ply_rate": sum(game.outcome.termination == "max_plies" for game in games)
        / len(games),
        "threefold_rate": sum(
            game.outcome.termination == "threefold_repetition" for game in games
        )
        / len(games),
        "mean_plies": sum(game.plies for game in games) / len(games),
    }


def summarize_control(games: tuple[QualificationGame, ...]) -> dict[str, float | int]:
    if not games or any(game.candidate_score is not None for game in games):
        raise ValueError("control summary requires raw-versus-raw games")
    return {
        "games": len(games),
        "max_ply_rate": sum(game.outcome.termination == "max_plies" for game in games)
        / len(games),
        "threefold_rate": sum(
            game.outcome.termination == "threefold_repetition" for game in games
        )
        / len(games),
        "mean_plies": sum(game.plies for game in games) / len(games),
    }


def evaluate_gate(
    arms: dict[int, dict[str, object]],
    control: dict[str, float | int],
    tactical: dict[str, object],
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for budget in (128, 256):
        if float(arms[budget]["score_rate"]) <= 0.55:
            reasons.append(f"{budget} search score did not exceed 0.55")
    interval = arms[256]["score_interval"]
    if not isinstance(interval, dict) or float(interval["low"]) <= 0.50:
        reasons.append("256 search paired lower bound did not exceed 0.50")
    if float(arms[256]["score_rate"]) < float(arms[128]["score_rate"]) - 0.02:
        reasons.append("256 search regressed versus 128")
    sweeps = {int(row["budget"]): row for row in tactical["budgets"]}  # type: ignore[index]
    raw_solved = int(tactical["raw"]["solved"])  # type: ignore[index]
    if int(sweeps[256]["solved"]) < raw_solved + 2:
        reasons.append("256 tactical solve gain was below two cases")
    if int(sweeps[256]["solved"]) < int(sweeps[128]["solved"]):
        reasons.append("256 tactical solve count regressed versus 128")
    solved_128 = {row["case"] for row in sweeps[128]["cases"] if row["solved"]}
    solved_256 = {row["case"] for row in sweeps[256]["cases"] if row["solved"]}
    if solved_128 - solved_256:
        reasons.append("256 search lost a tactic solved at 128")
    if float(arms[256]["decisive_score"]) < 0.50:
        reasons.append("256 decisive score was below 0.50")
    for metric in ("max_ply_rate", "threefold_rate"):
        if float(arms[256][metric]) > float(control[metric]) + 0.10:
            reasons.append(f"256 {metric} exceeded raw control margin")
    return not reasons, tuple(reasons)


def _inference_payload(statistics: BatchStatistics, elapsed: float) -> dict[str, float | int]:
    return {
        **asdict(statistics),
        "average_batch_size": statistics.average_batch_size,
        "average_queue_wait_ms": statistics.average_queue_wait_ms,
        "positions_per_second": statistics.positions / max(elapsed, 1e-9),
    }


def run_system_teacher_qualification(config: SystemTeacherConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"qualification output already exists: {config.output_dir}")
    actual_hash = _sha256(config.model_path)
    if actual_hash != config.model_sha256:
        raise ValueError("system teacher model checksum mismatch")
    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=config.trunk_channels,
            residual_blocks=config.residual_blocks,
            policy_channels=config.policy_channels,
            value_channels=config.value_channels,
            value_hidden=config.value_hidden,
        )
    )
    network.load_weights(str(config.model_path))
    rules = PythonChessRules()
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers),
        max_wait_seconds=config.inference_wait_seconds,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    raw = RawPolicy(evaluator)
    searches = {
        budget: MCTS(
            evaluator,
            rules=rules,
            config=SearchConfig(simulations=budget, dirichlet_fraction=0.0),
        )
        for budget in config.budgets
    }
    openings = _openings(
        rules, count=config.opening_pairs, plies=config.opening_plies, seed=config.seed
    )
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.EVALUATION,
        mode_detail="ESAS system teacher qualification · tactical diagnostics",
        run_id="esas-system-teacher-20260828-01",
        active_games=0,
        completed_games=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    tactical = run_tactical_sweep(
        evaluator,
        rules=rules,
        budgets=config.budgets,
        workers=min(config.workers, 8),
        seed=config.seed,
    )

    def run_arm(candidate: RawPolicy | MCTS | None, label: str) -> tuple[QualificationGame, ...]:
        tasks = [
            (pair_index, side, state, moves)
            for pair_index, (state, moves) in enumerate(openings)
            for side in (Side.WHITE, Side.BLACK)
        ]
        completed = 0
        lock = threading.Lock()

        def play(task: tuple[int, Side, ChessState, tuple[str, ...]]) -> QualificationGame:
            nonlocal completed, snapshot
            pair_index, side, state, moves = task
            white = raw if candidate is None or side is Side.BLACK else candidate
            black = raw if candidate is None or side is Side.WHITE else candidate
            game = _play_game(
                white,
                black,
                rules,
                state,
                pair_index=pair_index,
                candidate_side=None if candidate is None else side,
                opening_moves=moves,
                max_plies=config.max_plies,
            )
            with lock:
                completed += 1
                elapsed = time.perf_counter() - started
                snapshot = replace(
                    snapshot,
                    updated_at=datetime.now(UTC).isoformat(),
                    mode_detail=f"ESAS system teacher · {label} · {completed}/{len(tasks)} games",
                    session_elapsed_seconds=elapsed,
                    active_games=len(tasks) - completed,
                    completed_games=completed,
                    games_per_hour=completed / max(elapsed, 1e-9) * 3600.0,
                    positions_per_second=batcher.statistics.positions / max(elapsed, 1e-9),
                )
                store.write_atomic(snapshot)
            return game

        with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
            return tuple(pool.map(play, tasks))

    try:
        control_games = run_arm(None, "raw control")
        control = summarize_control(control_games)
        arms: dict[int, dict[str, object]] = {}
        serialized_games: dict[str, list[dict[str, object]]] = {
            "raw": [asdict(game) for game in control_games]
        }
        for budget, search in searches.items():
            arm_started = time.perf_counter()
            games = run_arm(search, f"{budget} simulations")
            summary = summarize_games(
                games,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.seed + budget,
            )
            summary["elapsed_seconds"] = time.perf_counter() - arm_started
            arms[budget] = summary
            serialized_games[str(budget)] = [asdict(game) for game in games]
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    passed, reasons = evaluate_gate(arms, control, tactical)
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "passed": passed,
            "reasons": reasons,
            "learner_latest_authorized": passed,
            "generation_authorized": False,
            "promotion_authorized": False,
            "config": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "model_path": str(config.model_path),
                "telemetry_path": str(config.telemetry_path),
            },
            "model_sha256": actual_hash,
            "tactical": tactical,
            "raw_control": control,
            "arms": arms,
            "games": serialized_games,
            "elapsed_seconds": elapsed,
            "inference": _inference_payload(batcher.statistics, elapsed),
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "ESAS system teacher qualified · learner/latest implementation authorized"
            if passed
            else "ESAS system teacher rejected · learner remains blocked"
        ),
        active_games=0,
    )
    store.write_atomic(snapshot)
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--opening-pairs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_system_teacher_qualification(
        SystemTeacherConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            telemetry_path=arguments.telemetry,
            opening_pairs=arguments.opening_pairs,
            workers=arguments.workers,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
