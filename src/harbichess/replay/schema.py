"""Versioned replay records derived from validated self-play games."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from harbichess.chess.actions import ACTION_SCHEMA_VERSION, POLICY_SIZE, move_to_action
from harbichess.chess.encoding import ENCODER_SCHEMA_VERSION
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState, Side
from harbichess.selfplay.game import SelfPlayGame, SelfPlaySample

REPLAY_SCHEMA_VERSION = 2
TARGET_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    game_id: str
    game_index: int
    seed: int
    ply: int
    root_fen: str
    moves: tuple[str, ...]
    side_to_move: Side
    policy: tuple[tuple[int, float], ...]
    selected_action: int
    root_value: float
    outcome_value: int
    repetition_redirected: bool

    def __post_init__(self) -> None:
        if not self.game_id or self.game_index < 0 or self.seed < 0 or self.ply != len(self.moves):
            raise ValueError("replay identity and ply history are inconsistent")
        if not math.isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError("root value must be finite and between -1 and 1")
        if self.outcome_value not in (-1, 0, 1):
            raise ValueError("outcome value must be -1, 0, or 1")
        indices = [action for action, _ in self.policy]
        probabilities = [probability for _, probability in self.policy]
        if (
            not self.policy
            or len(indices) != len(set(indices))
            or any(not 0 <= action < POLICY_SIZE for action in indices)
            or any(not math.isfinite(value) or value < 0 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("replay policy must be unique, legal-sized, finite, and normalized")
        if not any(
            action == self.selected_action and probability > 0
            for action, probability in self.policy
        ):
            raise ValueError("selected action must have positive replay-policy mass")

    @property
    def state(self) -> ChessState:
        return ChessState(self.root_fen, tuple(ChessMove(move) for move in self.moves))

    def validate_rules(self, rules: PythonChessRules) -> None:
        board = rules.board(self.state)
        expected_side = Side.WHITE if board.turn else Side.BLACK
        if self.side_to_move is not expected_side:
            raise ValueError("replay side-to-move does not match reconstructed state")
        legal_actions = {move_to_action(board, move) for move in board.legal_moves}
        if any(action not in legal_actions for action, _ in self.policy):
            raise ValueError("replay policy contains an illegal action")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayRecord:
        parsed = dict(data)
        parsed["moves"] = tuple(parsed["moves"])
        parsed["side_to_move"] = Side(parsed["side_to_move"])
        parsed["policy"] = tuple(
            (int(action), float(probability)) for action, probability in parsed["policy"]
        )
        return cls(**parsed)


def record_from_sample(
    game: SelfPlayGame,
    sample: SelfPlaySample,
    *,
    run_id: str,
    rules: PythonChessRules,
) -> ReplayRecord:
    board = rules.board(sample.state)
    expected_side = Side.WHITE if board.turn else Side.BLACK
    if sample.side_to_move is not expected_side:
        raise ValueError("self-play sample perspective does not match its state")
    policy = tuple(
        sorted(
            (move_to_action(board, board.parse_uci(move.uci)), probability)
            for move, probability in sample.visit_policy
        )
    )
    selected = move_to_action(board, board.parse_uci(sample.selected_move.uci))
    record = ReplayRecord(
        game_id=f"{run_id}-{game.game_index:012d}",
        game_index=game.game_index,
        seed=game.seed,
        ply=sample.state.ply,
        root_fen=sample.state.root_fen,
        moves=tuple(move.uci for move in sample.state.moves),
        side_to_move=sample.side_to_move,
        policy=policy,
        selected_action=selected,
        root_value=sample.root_value,
        outcome_value=sample.outcome_value,
        repetition_redirected=sample.repetition_redirected,
    )
    record.validate_rules(rules)
    return record


def records_from_game(
    game: SelfPlayGame,
    *,
    run_id: str,
    rules: PythonChessRules | None = None,
) -> tuple[ReplayRecord, ...]:
    engine = rules or PythonChessRules()
    return tuple(
        record_from_sample(game, sample, run_id=run_id, rules=engine) for sample in game.samples
    )


SCHEMA_VERSIONS = {
    "replay": REPLAY_SCHEMA_VERSION,
    "encoder": ENCODER_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "target": TARGET_SCHEMA_VERSION,
}
