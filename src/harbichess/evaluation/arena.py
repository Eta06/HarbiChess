"""Color-balanced DEVIR candidate arena with live dashboard telemetry."""

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

import mlx.core as mx

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState, GameOutcome, Side, TerminalResult
from harbichess.dashboard.state import (
    HistoryPoint,
    LiveGame,
    RunMode,
    SnapshotStore,
)
from harbichess.evaluation.quality import ArenaQuality, estimate_arena_quality
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import ShardMetadata, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.continuation import transform_repetition_target
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig, SearchResult


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    arena_id: str
    ocak_result: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    opening_pairs: int = 8
    opening_plies: int = 4
    simulations: int = 4
    max_plies: int = 192
    workers: int = 16
    seed: int = 20260825
    minimum_promotion_games: int = 200
    promotion_elo: float = 0.0
    candidate_checkpoint: str | None = None
    inference_wait_seconds: float = 0.00025

    def __post_init__(self) -> None:
        if not self.arena_id or Path(self.arena_id).name != self.arena_id:
            raise ValueError("arena_id must be one safe path segment")
        counts = (
            self.opening_pairs,
            self.simulations,
            self.max_plies,
            self.workers,
            self.minimum_promotion_games,
        )
        if any(value <= 0 for value in counts) or self.opening_plies < 0:
            raise ValueError("arena counts must be positive and opening plies non-negative")
        if self.seed < 0:
            raise ValueError("arena seed must be non-negative")
        if self.inference_wait_seconds < 0:
            raise ValueError("inference_wait_seconds must be non-negative")


def _select_checkpoint(
    ocak: dict[str, object],
    checkpoint_id: str | None,
) -> dict[str, object]:
    selected = ocak["checkpoint"]
    if checkpoint_id is None:
        return selected
    candidates = ocak.get("validation_checkpoints", ())
    for candidate in candidates:
        if candidate["manifest"]["checkpoint_id"] == checkpoint_id:
            return candidate
    if selected["manifest"]["checkpoint_id"] == checkpoint_id:
        return selected
    raise ValueError(f"unknown validation checkpoint: {checkpoint_id}")


@dataclass(frozen=True, slots=True)
class ArenaGame:
    game_id: str
    pair_index: int
    candidate_side: Side
    opening_moves: tuple[str, ...]
    final_state: ChessState
    outcome: GameOutcome
    candidate_score: float
    last_search: SearchResult | None
    avoidable_threefold: bool
    continuation_roots: tuple[ContinuationRoot, ...]


@dataclass(frozen=True, slots=True)
class ContinuationRoot:
    state: ChessState
    side_to_move: Side
    policy: tuple[tuple[ChessMove, float], ...]
    selected_move: ChessMove
    root_value: float
    repeating_policy_mass: float
    source_model: str


@dataclass(frozen=True, slots=True)
class ArenaResult:
    arena_id: str
    candidate_checkpoint: str
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float
    elo_delta: float
    elo_low: float | None
    elo_high: float | None
    promotion_ready: bool
    elapsed_seconds: float
    result_path: str
    continuation_replay_samples: int
    continuation_replay_path: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
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


def _candidate_score(outcome: GameOutcome, candidate_side: Side) -> float:
    value = outcome.value_for(candidate_side)
    return 1.0 if value > 0 else 0.0 if value < 0 else 0.5


def _avoidable_threefold(
    rules: PythonChessRules,
    state: ChessState,
    selected_move: ChessMove,
) -> bool:
    selected_state = rules.apply(state, selected_move)
    selected_outcome = rules.outcome(selected_state, claim_draw=True)
    if selected_outcome is None or selected_outcome.termination != "threefold_repetition":
        return False
    for move in rules.legal_moves(state):
        if move == selected_move:
            continue
        alternative_outcome = rules.outcome(rules.apply(state, move), claim_draw=True)
        if alternative_outcome is None or alternative_outcome.termination != "threefold_repetition":
            return True
    return False


def play_arena_game(
    candidate: MCTS,
    champion: MCTS,
    rules: PythonChessRules,
    initial_state: ChessState,
    *,
    game_id: str,
    pair_index: int,
    candidate_side: Side,
    opening_moves: tuple[str, ...],
    max_plies: int,
) -> ArenaGame:
    state = initial_state
    last_search: SearchResult | None = None
    avoidable_threefold = False
    continuation_roots = []
    while True:
        outcome = rules.outcome(state, claim_draw=True)
        if outcome is not None:
            break
        if state.ply >= max_plies:
            outcome = GameOutcome(TerminalResult.DRAW, "max_plies")
            break
        side = rules.view(state).side_to_move
        search = candidate if side is candidate_side else champion
        last_search = search.search(state, rng=random.Random(0), add_root_noise=False)
        selected_move = last_search.select_move(temperature=0.0, rng=random.Random(0))
        avoidable_threefold = _avoidable_threefold(rules, state, selected_move)
        if avoidable_threefold:
            continuation = transform_repetition_target(
                last_search,
                rules,
                state,
                selected_move,
                temperature=0.0,
                value_tolerance=0.05,
                minimum_repeating_policy_mass=0.10,
                rng=random.Random(0),
            )
            if continuation.transformed:
                total_visits = sum(item.visits for item in continuation.policy_moves)
                continuation_roots.append(
                    ContinuationRoot(
                        state=state,
                        side_to_move=side,
                        policy=tuple(
                            (item.move, item.visits / total_visits)
                            for item in continuation.policy_moves
                        ),
                        selected_move=continuation.selected_move,
                        root_value=last_search.root_value,
                        repeating_policy_mass=continuation.repeating_policy_mass,
                        source_model=("candidate" if side is candidate_side else "champion"),
                    )
                )
        state = rules.apply(state, selected_move)
    return ArenaGame(
        game_id=game_id,
        pair_index=pair_index,
        candidate_side=candidate_side,
        opening_moves=opening_moves,
        final_state=state,
        outcome=outcome,
        candidate_score=_candidate_score(outcome, candidate_side),
        last_search=last_search,
        avoidable_threefold=avoidable_threefold,
        continuation_roots=tuple(continuation_roots),
    )


def _continuation_records(
    games: tuple[ArenaGame, ...],
    *,
    arena_id: str,
    seed: int,
    rules: PythonChessRules,
) -> tuple[ReplayRecord, ...]:
    records = []
    for game_index, game in enumerate(games):
        for root in game.continuation_roots:
            board = rules.board(root.state)
            records.append(
                ReplayRecord(
                    game_id=f"continuation-{arena_id}-{game.game_id}",
                    game_index=game_index,
                    seed=seed,
                    ply=root.state.ply,
                    root_fen=root.state.root_fen,
                    moves=tuple(move.uci for move in root.state.moves),
                    side_to_move=root.side_to_move,
                    policy=tuple(
                        sorted(
                            (
                                move_to_action(board, board.parse_uci(move.uci)),
                                probability,
                            )
                            for move, probability in root.policy
                        )
                    ),
                    selected_action=move_to_action(
                        board,
                        board.parse_uci(root.selected_move.uci),
                    ),
                    root_value=root.root_value,
                    outcome_value=(
                        None
                        if game.outcome.termination == "max_plies"
                        else game.outcome.value_for(root.side_to_move)
                    ),
                    repetition_redirected=True,
                )
            )
    return tuple(records)


def _openings(
    rules: PythonChessRules,
    *,
    count: int,
    plies: int,
    seed: int,
) -> tuple[tuple[ChessState, tuple[str, ...]], ...]:
    openings = []
    seen: set[tuple[str, ...]] = set()
    candidate_index = 0
    while len(openings) < count:
        rng = random.Random(f"{seed}:{candidate_index}")
        candidate_index += 1
        state = rules.initial_state()
        moves = []
        for _ in range(plies):
            legal = rules.legal_moves(state)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move.uci)
            state = rules.apply(state, move)
        opening = tuple(moves)
        if opening in seen:
            continue
        seen.add(opening)
        openings.append((state, opening))
    return tuple(openings)


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=int(payload["trunk_channels"]),
        residual_blocks=int(payload["residual_blocks"]),
        policy_channels=int(payload["policy_channels"]),
        value_channels=int(payload["value_channels"]),
        value_hidden=int(payload["value_hidden"]),
    )


def _live_game(game: ArenaGame, rules: PythonChessRules) -> LiveGame:
    search = game.last_search
    moves = (
        ()
        if search is None
        else tuple(
            (item.move.uci, item.visits / max(1, search.simulations)) for item in search.moves[:3]
        )
    )
    value = 0.0 if search is None else search.root_value
    return LiveGame(
        game_id=game.game_id,
        white="Candidate" if game.candidate_side is Side.WHITE else "Baseline",
        black="Candidate" if game.candidate_side is Side.BLACK else "Baseline",
        fen=rules.board(game.final_state).fen(),
        last_move=game.final_state.moves[-1].uci if game.final_state.moves else "",
        ply=game.final_state.ply,
        top_moves=moves,
        wdl=(max(0.0, value), max(0.0, 1.0 - abs(value)), max(0.0, -value)),
    )


def run_devir_arena(config: ArenaConfig) -> ArenaResult:
    ocak = json.loads(config.ocak_result.read_text(encoding="utf-8"))
    if not ocak.get("passed"):
        raise ValueError("DEVIR arena requires a passed OCAK result")
    selected_checkpoint = _select_checkpoint(ocak, config.candidate_checkpoint)
    network_config = _network_config(ocak["config"])
    run_root = config.ocak_result.parent
    arena_root = run_root / "arena" / config.arena_id
    if arena_root.exists():
        raise FileExistsError(f"arena already exists: {arena_root}")
    arena_root.mkdir(parents=True)

    candidate_path = (
        Path(selected_checkpoint["path"]) / selected_checkpoint["manifest"]["model_file"]
    )
    candidate = HarbiChessNetwork(network_config)
    candidate.load_weights(str(candidate_path))
    champion = HarbiChessNetwork(network_config)
    baseline = ocak.get("baseline")
    if baseline is not None:
        champion_path = Path(baseline["path"])
        if _sha256(champion_path) != baseline["model_sha256"]:
            raise ValueError("OCAK baseline model checksum mismatch")
        champion.load_weights(str(champion_path))
    else:
        mx.random.seed(int(ocak["config"]["run_seed"]))
        champion = HarbiChessNetwork(network_config)
        champion_path = arena_root / "baseline-initial.safetensors"
        champion.save_weights(str(champion_path))

    rules = PythonChessRules()
    candidate_batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(candidate),
        max_batch_size=min(128, config.workers),
        max_wait_seconds=config.inference_wait_seconds,
    )
    champion_batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(champion),
        max_batch_size=min(128, config.workers),
        max_wait_seconds=config.inference_wait_seconds,
    )
    candidate_search = MCTS(
        NeuralPositionEvaluator(candidate_batcher, rules=rules),
        rules=rules,
        config=SearchConfig(simulations=config.simulations, dirichlet_fraction=0.0),
    )
    champion_search = MCTS(
        NeuralPositionEvaluator(champion_batcher, rules=rules),
        rules=rules,
        config=SearchConfig(simulations=config.simulations, dirichlet_fraction=0.0),
    )
    openings = _openings(
        rules,
        count=config.opening_pairs,
        plies=config.opening_plies,
        seed=config.seed,
    )
    tasks = [
        (pair_index, candidate_side, state, moves)
        for pair_index, (state, moves) in enumerate(openings)
        for candidate_side in (Side.WHITE, Side.BLACK)
    ]
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=_now(),
        mode=RunMode.EVALUATION,
        mode_detail=f"DEVIR color-balanced arena · 0/{len(tasks)} games",
        candidate_checkpoint=selected_checkpoint["manifest"]["checkpoint_id"],
        pilot_arena_selected_step=selected_checkpoint["manifest"].get("training_step", 0),
        pilot_arena_selection_reason="explicit same-generation arena checkpoint",
        active_games=len(tasks),
        completed_games=0,
        arena_games=0,
        arena_wins=0,
        arena_draws=0,
        arena_losses=0,
        arena_decisive_games=0,
        arena_threefold_repetitions=0,
        arena_avoidable_threefold_repetitions=0,
        arena_continuation_replay_samples=0,
        arena_max_ply_draws=0,
        arena_other_draws=0,
        arena_score_rate=0.5,
        arena_elo_delta=None,
        arena_elo_low=None,
        arena_elo_high=None,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    base_session_seconds = snapshot.session_elapsed_seconds
    started = time.perf_counter()
    lock = threading.Lock()
    games: list[ArenaGame] = []

    def publish(game: ArenaGame) -> None:
        nonlocal snapshot
        with lock:
            games.append(game)
            wins = sum(item.candidate_score == 1.0 for item in games)
            draws = sum(item.candidate_score == 0.5 for item in games)
            losses = sum(item.candidate_score == 0.0 for item in games)
            threefold = sum(item.outcome.termination == "threefold_repetition" for item in games)
            avoidable_threefold = sum(item.avoidable_threefold for item in games)
            max_ply = sum(item.outcome.termination == "max_plies" for item in games)
            other_draws = draws - threefold - max_ply
            quality = estimate_arena_quality(
                wins,
                draws,
                losses,
                minimum_games=config.minimum_promotion_games,
                promotion_elo=config.promotion_elo,
            )
            elapsed = time.perf_counter() - started
            snapshot = replace(
                snapshot,
                updated_at=_now(),
                mode_detail=f"DEVIR color-balanced arena · {len(games)}/{len(tasks)} games",
                session_elapsed_seconds=base_session_seconds + elapsed,
                active_games=len(tasks) - len(games),
                completed_games=len(games),
                arena_games=quality.games,
                arena_wins=quality.wins,
                arena_draws=quality.draws,
                arena_losses=quality.losses,
                arena_decisive_games=wins + losses,
                arena_threefold_repetitions=threefold,
                arena_avoidable_threefold_repetitions=avoidable_threefold,
                arena_continuation_replay_samples=sum(
                    len(item.continuation_roots) for item in games
                ),
                arena_max_ply_draws=max_ply,
                arena_other_draws=other_draws,
                arena_score_rate=quality.score_rate,
                arena_elo_delta=quality.elo_delta,
                arena_elo_low=quality.elo_low,
                arena_elo_high=quality.elo_high,
                promotion_ready=quality.promotion_ready,
                games_per_hour=len(games) / max(elapsed, 1e-9) * 3600.0,
                live_game=_live_game(game, rules),
            )
            store.write_atomic(snapshot)

    def play(task: tuple[int, Side, ChessState, tuple[str, ...]]) -> ArenaGame:
        pair_index, candidate_side, state, moves = task
        game = play_arena_game(
            candidate_search,
            champion_search,
            rules,
            state,
            game_id=f"{config.arena_id}-{pair_index:04d}-{candidate_side.value}",
            pair_index=pair_index,
            candidate_side=candidate_side,
            opening_moves=moves,
            max_plies=config.max_plies,
        )
        publish(game)
        return game

    try:
        with ThreadPoolExecutor(max_workers=min(config.workers, len(tasks))) as pool:
            completed = tuple(pool.map(play, tasks))
    finally:
        candidate_batcher.close()
        champion_batcher.close()

    candidate_inference = candidate_batcher.statistics
    champion_inference = champion_batcher.statistics

    wins = sum(game.candidate_score == 1.0 for game in completed)
    draws = sum(game.candidate_score == 0.5 for game in completed)
    losses = sum(game.candidate_score == 0.0 for game in completed)
    quality: ArenaQuality = estimate_arena_quality(
        wins,
        draws,
        losses,
        minimum_games=config.minimum_promotion_games,
        promotion_elo=config.promotion_elo,
    )
    elapsed = time.perf_counter() - started
    continuation_records = _continuation_records(
        completed,
        arena_id=config.arena_id,
        seed=config.seed,
        rules=rules,
    )
    continuation_path = arena_root / "continuation-replay.jsonl.gz"
    continuation_header = (
        write_shard_atomic(
            continuation_path,
            continuation_records,
            ShardMetadata(
                run_id=config.arena_id,
                generation=0,
                source_checkpoint=selected_checkpoint["manifest"]["checkpoint_id"],
                source_commit=_source_commit(),
                created_at=_now(),
                split=ReplaySplit.TRAIN,
            ),
        )
        if continuation_records
        else None
    )
    result_path = arena_root / "result.json"
    _atomic_json(
        result_path,
        {
            "arena_id": config.arena_id,
            "created_at": _now(),
            "candidate_checkpoint": selected_checkpoint["manifest"]["checkpoint_id"],
            "candidate_source_commit": ocak["source_commit"],
            "candidate_model_sha256": _sha256(candidate_path),
            "baseline_model": str(champion_path),
            "baseline_model_sha256": _sha256(champion_path),
            "config": {
                **asdict(config),
                "ocak_result": str(config.ocak_result),
                "telemetry_path": str(config.telemetry_path),
            },
            "quality": asdict(quality),
            "elapsed_seconds": elapsed,
            "inference": {
                "candidate": {
                    **asdict(candidate_inference),
                    "average_batch_size": candidate_inference.average_batch_size,
                    "average_queue_wait_ms": candidate_inference.average_queue_wait_ms,
                },
                "champion": {
                    **asdict(champion_inference),
                    "average_batch_size": champion_inference.average_batch_size,
                    "average_queue_wait_ms": champion_inference.average_queue_wait_ms,
                },
            },
            "continuation_replay": (
                {
                    "path": str(continuation_path),
                    "source_commit": continuation_header.source_commit,
                    "header": asdict(continuation_header),
                    "candidate_roots": sum(
                        root.source_model == "candidate"
                        for game in completed
                        for root in game.continuation_roots
                    ),
                    "champion_roots": sum(
                        root.source_model == "champion"
                        for game in completed
                        for root in game.continuation_roots
                    ),
                    "mean_repeating_policy_mass": sum(
                        root.repeating_policy_mass
                        for game in completed
                        for root in game.continuation_roots
                    )
                    / len(continuation_records),
                }
                if continuation_header is not None
                else None
            ),
            "games": [
                {
                    "game_id": game.game_id,
                    "pair_index": game.pair_index,
                    "candidate_side": game.candidate_side,
                    "opening_moves": game.opening_moves,
                    "result": game.outcome.result,
                    "termination": game.outcome.termination,
                    "avoidable_threefold": game.avoidable_threefold,
                    "plies": game.final_state.ply,
                    "candidate_score": game.candidate_score,
                }
                for game in completed
            ],
        },
    )
    point = HistoryPoint(
        training_step=snapshot.training_step,
        training_elapsed_seconds=snapshot.training_elapsed_seconds,
        lifetime_games=snapshot.lifetime_games,
        total_loss=snapshot.total_loss,
        elo_delta=quality.elo_delta,
        elo_low=quality.elo_low,
        elo_high=quality.elo_high,
        games_per_hour=len(completed) / max(elapsed, 1e-9) * 3600.0,
        positions_per_second=snapshot.positions_per_second,
        policy_loss=snapshot.policy_loss,
        value_loss=snapshot.value_loss,
        validation_loss=snapshot.pilot_final_validation_loss,
    )
    snapshot = replace(
        snapshot,
        updated_at=_now(),
        mode=RunMode.IDLE,
        mode_detail=(
            "DEVIR arena passed promotion confidence"
            if quality.promotion_ready
            else "DEVIR arena complete · champion unchanged"
        ),
        active_games=0,
        history=(*snapshot.history, point)[-240:],
    )
    store.write_atomic(snapshot)
    return ArenaResult(
        arena_id=config.arena_id,
        candidate_checkpoint=selected_checkpoint["manifest"]["checkpoint_id"],
        games=quality.games,
        wins=quality.wins,
        draws=quality.draws,
        losses=quality.losses,
        score_rate=quality.score_rate,
        elo_delta=quality.elo_delta,
        elo_low=quality.elo_low,
        elo_high=quality.elo_high,
        promotion_ready=quality.promotion_ready,
        elapsed_seconds=elapsed,
        result_path=str(result_path),
        continuation_replay_samples=len(continuation_records),
        continuation_replay_path=(str(continuation_path) if continuation_records else None),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-id", required=True)
    parser.add_argument("--ocak-result", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--opening-pairs", type=int, default=8)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--inference-wait-ms", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_devir_arena(
        ArenaConfig(
            arena_id=arguments.arena_id,
            ocak_result=arguments.ocak_result,
            telemetry_path=arguments.telemetry,
            opening_pairs=arguments.opening_pairs,
            opening_plies=arguments.opening_plies,
            simulations=arguments.simulations,
            max_plies=arguments.max_plies,
            workers=arguments.workers,
            seed=arguments.seed,
            candidate_checkpoint=arguments.candidate_checkpoint,
            inference_wait_seconds=arguments.inference_wait_ms / 1_000,
        )
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
