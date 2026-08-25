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
TARGET_SCHEMA_VERSION = 4
SUPPORTED_TARGET_SCHEMA_VERSIONS = frozenset({3, TARGET_SCHEMA_VERSION})


@dataclass(frozen=True, slots=True)
class BranchValueEstimate:
    action: int
    move: str
    samples: int
    mean_value: float
    standard_error: float
    lower_confidence_bound: float
    upper_confidence_bound: float

    def __post_init__(self) -> None:
        values = (
            self.mean_value,
            self.standard_error,
            self.lower_confidence_bound,
            self.upper_confidence_bound,
        )
        if not 0 <= self.action < POLICY_SIZE or not self.move or self.samples <= 1:
            raise ValueError("branch evidence identity and samples are invalid")
        if any(not math.isfinite(value) for value in values) or self.standard_error < 0:
            raise ValueError("branch evidence values must be finite")
        if not -1.0 <= self.mean_value <= 1.0:
            raise ValueError("branch mean value must be between -1 and 1")
        if not -1.0 <= self.lower_confidence_bound <= self.upper_confidence_bound <= 1.0:
            raise ValueError("branch confidence bounds must be ordered within [-1, 1]")


@dataclass(frozen=True, slots=True)
class ContinuationEvidence:
    method_version: int
    confidence_level: float
    branch_searches: int
    simulations_per_search: int
    repeat_value: float
    repeat_actions: tuple[int, ...]
    branches: tuple[BranchValueEstimate, ...]
    qualified_actions: tuple[int, ...]
    source_model_sha256: str

    def __post_init__(self) -> None:
        if (
            self.method_version <= 0
            or self.branch_searches <= 1
            or self.simulations_per_search <= 0
        ):
            raise ValueError("continuation evidence search configuration is invalid")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("continuation evidence confidence level must be in (0, 1)")
        if not math.isfinite(self.repeat_value) or not -1.0 <= self.repeat_value <= 1.0:
            raise ValueError("continuation repeat value must be finite and bounded")
        if not self.repeat_actions or len(self.repeat_actions) != len(set(self.repeat_actions)):
            raise ValueError("continuation repeat actions must be unique and non-empty")
        if not self.branches or len({branch.action for branch in self.branches}) != len(
            self.branches
        ):
            raise ValueError("continuation branch actions must be unique and non-empty")
        branch_actions = {branch.action for branch in self.branches}
        if (
            len(self.qualified_actions) != len(set(self.qualified_actions))
            or not set(self.qualified_actions) <= branch_actions
            or set(self.qualified_actions) & set(self.repeat_actions)
        ):
            raise ValueError("qualified actions must be unique evaluated non-repeat branches")
        if len(self.source_model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_model_sha256.lower()
        ):
            raise ValueError("continuation evidence model hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContinuationEvidence:
        parsed = dict(data)
        parsed["repeat_actions"] = tuple(int(action) for action in parsed["repeat_actions"])
        parsed["branches"] = tuple(BranchValueEstimate(**branch) for branch in parsed["branches"])
        parsed["qualified_actions"] = tuple(int(action) for action in parsed["qualified_actions"])
        return cls(**parsed)


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
    continuation_evidence: ContinuationEvidence | None = None

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
        if self.continuation_evidence is not None:
            qualified = set(self.continuation_evidence.qualified_actions)
            if not qualified or set(indices) != qualified or self.selected_action not in qualified:
                raise ValueError("confidence-gated policy must exactly match qualified actions")

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
        if self.continuation_evidence is not None:
            evidence_actions = {
                *self.continuation_evidence.repeat_actions,
                *(branch.action for branch in self.continuation_evidence.branches),
            }
            if not evidence_actions <= legal_actions:
                raise ValueError("continuation evidence contains an illegal action")

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
        evidence = parsed.get("continuation_evidence")
        parsed["continuation_evidence"] = (
            ContinuationEvidence.from_dict(evidence) if evidence is not None else None
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
