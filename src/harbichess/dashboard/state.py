"""Low-frequency telemetry snapshot shared with the dashboard process."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from harbichess.evaluation.quality import estimate_arena_quality

MAX_HISTORY_POINTS = 240


class RunMode(StrEnum):
    SELF_PLAY = "SELF_PLAY"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    CHECKPOINTING = "CHECKPOINTING"
    PAUSED = "PAUSED"
    IDLE = "IDLE"


@dataclass(frozen=True, slots=True)
class LiveGame:
    game_id: str = ""
    white: str = "HarbiChess"
    black: str = "HarbiChess"
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    last_move: str = ""
    ply: int = 0
    top_moves: tuple[tuple[str, float], ...] = ()
    wdl: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    training_step: int
    training_elapsed_seconds: float
    lifetime_games: int
    total_loss: float | None
    elo_delta: float | None
    elo_low: float | None
    elo_high: float | None
    games_per_hour: float
    positions_per_second: float


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    schema_version: int
    updated_at: str
    mode: RunMode
    mode_detail: str
    run_id: str
    session_elapsed_seconds: float
    training_elapsed_seconds: float
    active_checkpoint: str
    promoted_checkpoint: str
    candidate_checkpoint: str
    source_commit: str
    training_step: int
    lifetime_games: int
    run_games: int
    generation_games: int
    active_games: int
    completed_games: int
    failed_games: int
    games_per_hour: float
    positions_per_second: float
    neural_evals_per_second: float
    mcts_nodes_per_second: float
    inference_batch_size: int
    inference_queue_depth: int
    replay_samples: int
    replay_capacity: int
    policy_loss: float | None
    value_loss: float | None
    total_loss: float | None
    learning_rate: float | None
    demo: bool = False
    live_game: LiveGame = field(default_factory=LiveGame)
    arena_games: int = 0
    arena_wins: int = 0
    arena_draws: int = 0
    arena_losses: int = 0
    arena_score_rate: float = 0.5
    arena_elo_delta: float | None = None
    arena_elo_low: float | None = None
    arena_elo_high: float | None = None
    promotion_ready: bool = False
    history: tuple[HistoryPoint, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> DashboardSnapshot:
        data: dict[str, Any] = json.loads(payload)
        data["mode"] = RunMode(data["mode"])
        game = data.get("live_game", {})
        game["top_moves"] = tuple(tuple(item) for item in game.get("top_moves", ()))
        game["wdl"] = tuple(game.get("wdl", (0.0, 1.0, 0.0)))
        data["live_game"] = LiveGame(**game)
        data["history"] = tuple(
            HistoryPoint(**point) for point in data.get("history", ())[-MAX_HISTORY_POINTS:]
        )
        return cls(**data)

    def append_history(self, point: HistoryPoint) -> DashboardSnapshot:
        return replace(self, history=(*self.history, point)[-MAX_HISTORY_POINTS:])


def empty_snapshot() -> DashboardSnapshot:
    return DashboardSnapshot(
        schema_version=2,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail="Waiting for the first HarbiChess run",
        run_id="not-started",
        session_elapsed_seconds=0.0,
        training_elapsed_seconds=0.0,
        active_checkpoint="No checkpoint yet",
        promoted_checkpoint="None",
        candidate_checkpoint="None",
        source_commit="unknown",
        training_step=0,
        lifetime_games=0,
        run_games=0,
        generation_games=0,
        active_games=0,
        completed_games=0,
        failed_games=0,
        games_per_hour=0.0,
        positions_per_second=0.0,
        neural_evals_per_second=0.0,
        mcts_nodes_per_second=0.0,
        inference_batch_size=0,
        inference_queue_depth=0,
        replay_samples=0,
        replay_capacity=0,
        policy_loss=None,
        value_loss=None,
        total_loss=None,
        learning_rate=None,
    )


def demo_snapshot() -> DashboardSnapshot:
    snapshot = empty_snapshot()
    quality = estimate_arena_quality(122, 61, 37, minimum_games=200)
    history = tuple(
        HistoryPoint(
            training_step=4_000 + index * 960,
            training_elapsed_seconds=4_100 + index * 1_049,
            lifetime_games=1_061_000 + index * 14_893,
            total_loss=3.72 - index * 0.0537,
            elo_delta=-42.0 + index * 12.25,
            elo_low=-137.0 + index * 15.5,
            elo_high=53.0 + index * 9.0,
            games_per_hour=8_940.0 + index * 120.0,
            positions_per_second=70_100.0 + index * 568.0,
        )
        for index in range(16)
    )
    return DashboardSnapshot(
        **{
            **asdict(snapshot),
            "updated_at": datetime.now(UTC).isoformat(),
            "mode": RunMode.SELF_PLAY,
            "mode_detail": "Generating diverse games with generation 0007",
            "run_id": "demo-m4max",
            "session_elapsed_seconds": 28_472,
            "training_elapsed_seconds": 19_841,
            "active_checkpoint": "train-step-00018400",
            "promoted_checkpoint": "HarbiChess 0.06",
            "candidate_checkpoint": "candidate-0007",
            "source_commit": "e14d4b02c40",
            "training_step": 18_400,
            "lifetime_games": 1_284_392,
            "run_games": 84_392,
            "generation_games": 12_806,
            "active_games": 64,
            "completed_games": 84_328,
            "games_per_hour": 10_742.0,
            "positions_per_second": 78_617.0,
            "neural_evals_per_second": 76_904.0,
            "mcts_nodes_per_second": 141_220.0,
            "inference_batch_size": 96,
            "inference_queue_depth": 41,
            "replay_samples": 4_820_114,
            "replay_capacity": 8_000_000,
            "policy_loss": 2.184,
            "value_loss": 0.731,
            "total_loss": 2.915,
            "learning_rate": 0.0002,
            "demo": True,
            "arena_games": quality.games,
            "arena_wins": quality.wins,
            "arena_draws": quality.draws,
            "arena_losses": quality.losses,
            "arena_score_rate": quality.score_rate,
            "arena_elo_delta": quality.elo_delta,
            "arena_elo_low": quality.elo_low,
            "arena_elo_high": quality.elo_high,
            "promotion_ready": quality.promotion_ready,
            "history": history,
            "live_game": LiveGame(
                game_id="sp-0007-12806",
                fen="r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 4 10",
                last_move="e7e6",
                ply=18,
                top_moves=(("e3e4", 0.41), ("c3b5", 0.24), ("f1d1", 0.13)),
                wdl=(0.39, 0.46, 0.15),
            ),
        }
    )


class SnapshotStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> DashboardSnapshot:
        if not self.path.exists():
            return empty_snapshot()
        return DashboardSnapshot.from_json(self.path.read_text(encoding="utf-8"))

    def write_atomic(self, snapshot: DashboardSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(snapshot.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
