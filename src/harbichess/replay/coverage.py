"""Replay coverage and search-teacher telemetry qualification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import chess

from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.schema import ReplayRecord


@dataclass(frozen=True, slots=True)
class ReplayCoverageThresholds:
    minimum_samples: int = 8_000
    minimum_unique_position_ratio: float = 0.75
    minimum_opening_ratio: float = 0.05
    minimum_middlegame_ratio: float = 0.25
    minimum_endgame_ratio: float = 0.15
    minimum_tactical_ratio: float = 0.08
    minimum_quiet_ratio: float = 0.35
    minimum_value_bucket_ratio: float = 0.05
    minimum_outcome_bucket_ratio: float = 0.03
    minimum_material_signatures: int = 12
    minimum_position_signatures: int = 24
    minimum_teacher_telemetry_ratio: float = 0.99
    minimum_comparable_teacher_deltas: int = 100


@dataclass(frozen=True, slots=True)
class ReplayCoverageReport:
    samples: int
    games: int
    unique_position_ratio: float
    phase_counts: tuple[tuple[str, int], ...]
    tactical_counts: tuple[tuple[str, int], ...]
    value_bucket_counts: tuple[tuple[str, int], ...]
    outcome_counts: tuple[tuple[str, int], ...]
    material_balance_counts: tuple[tuple[str, int], ...]
    distinct_material_signatures: int
    distinct_position_signatures: int
    teacher_telemetry_samples: int
    teacher_telemetry_ratio: float
    teacher_action_changes: int
    teacher_action_change_ratio: float
    mean_teacher_policy_tv: float
    mean_teacher_policy_kl: float
    comparable_teacher_deltas: int
    positive_teacher_delta_ratio: float
    mean_teacher_delta: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _phase(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 80:
        return "middlegame"
    return "endgame"


def _value_bucket(value: float) -> str:
    if value > 0.25:
        return "winning"
    if value < -0.25:
        return "losing"
    return "drawing"


def _outcome_bucket(value: int | None) -> str:
    if value is None:
        return "unknown"
    return {-1: "losing", 0: "drawing", 1: "winning"}[value]


def _material_signature(board: chess.Board) -> tuple[int, ...]:
    return tuple(
        len(board.pieces(piece_type, color))
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in range(chess.PAWN, chess.KING + 1)
    )


def _material_balance(board: chess.Board) -> str:
    weights = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    balance = sum(
        weights[piece_type]
        * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
        for piece_type in weights
    )
    if balance >= 2:
        return "white-ahead"
    if balance <= -2:
        return "black-ahead"
    return "balanced"


def _position_signature(board: chess.Board) -> tuple[object, ...]:
    white_pawn_files = tuple(
        sorted(chess.square_file(square) for square in board.pieces(chess.PAWN, chess.WHITE))
    )
    black_pawn_files = tuple(
        sorted(chess.square_file(square) for square in board.pieces(chess.PAWN, chess.BLACK))
    )
    return (
        _material_signature(board),
        white_pawn_files,
        black_pawn_files,
        bool(board.castling_rights),
        board.turn,
    )


def _ratio(counts: Counter[str], key: str, total: int) -> float:
    return counts[key] / total if total else 0.0


def measure_replay_coverage(
    records: Sequence[ReplayRecord],
    *,
    thresholds: ReplayCoverageThresholds | None = None,
    rules: PythonChessRules | None = None,
) -> ReplayCoverageReport:
    if not records:
        raise ValueError("replay coverage requires at least one record")
    limits = thresholds or ReplayCoverageThresholds()
    engine = rules or PythonChessRules()
    phases: Counter[str] = Counter()
    tactics: Counter[str] = Counter()
    values: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    balances: Counter[str] = Counter()
    position_keys: set[str] = set()
    material_signatures: set[tuple[int, ...]] = set()
    position_signatures: set[tuple[object, ...]] = set()
    telemetry = []
    deltas = []

    for record in records:
        board = engine.board(record.state)
        selected = action_to_legal_move(board, record.selected_action)
        phases[_phase(record.ply)] += 1
        tactical = (
            board.is_check()
            or board.is_capture(selected)
            or selected.promotion is not None
            or board.gives_check(selected)
        )
        tactics["tactical" if tactical else "quiet"] += 1
        values[_value_bucket(record.root_value)] += 1
        outcomes[_outcome_bucket(record.outcome_value)] += 1
        balances[_material_balance(board)] += 1
        material_signatures.add(_material_signature(board))
        position_signatures.add(_position_signature(board))
        position_keys.add(" ".join(board.fen().split()[:4]))
        if record.raw_policy:
            telemetry.append(record)
        if record.teacher_search_value_delta is not None:
            deltas.append(record.teacher_search_value_delta)

    sample_count = len(records)
    telemetry_count = len(telemetry)
    action_changes = sum(record.teacher_argmax_changed is True for record in telemetry)
    positive_delta_ratio = sum(delta > 0 for delta in deltas) / len(deltas) if deltas else 0.0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    reasons = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            reasons.append(reason)

    require(sample_count >= limits.minimum_samples, "insufficient replay samples")
    require(
        len(position_keys) / sample_count >= limits.minimum_unique_position_ratio,
        "unique-position coverage below threshold",
    )
    for key, minimum in (
        ("opening", limits.minimum_opening_ratio),
        ("middlegame", limits.minimum_middlegame_ratio),
        ("endgame", limits.minimum_endgame_ratio),
    ):
        require(_ratio(phases, key, sample_count) >= minimum, f"{key} coverage below threshold")
    require(
        _ratio(tactics, "tactical", sample_count) >= limits.minimum_tactical_ratio,
        "tactical coverage below threshold",
    )
    require(
        _ratio(tactics, "quiet", sample_count) >= limits.minimum_quiet_ratio,
        "quiet coverage below threshold",
    )
    for key in ("winning", "drawing", "losing"):
        require(
            _ratio(values, key, sample_count) >= limits.minimum_value_bucket_ratio,
            f"teacher-value {key} coverage below threshold",
        )
        require(
            _ratio(outcomes, key, sample_count) >= limits.minimum_outcome_bucket_ratio,
            f"outcome {key} coverage below threshold",
        )
    require(
        len(material_signatures) >= limits.minimum_material_signatures,
        "material-signature coverage below threshold",
    )
    require(
        len(position_signatures) >= limits.minimum_position_signatures,
        "position-structure coverage below threshold",
    )
    require(
        telemetry_count / sample_count >= limits.minimum_teacher_telemetry_ratio,
        "teacher telemetry coverage below threshold",
    )
    require(
        len(deltas) >= limits.minimum_comparable_teacher_deltas,
        "too few comparable raw/teacher actions",
    )
    return ReplayCoverageReport(
        samples=sample_count,
        games=len({record.game_id for record in records}),
        unique_position_ratio=len(position_keys) / sample_count,
        phase_counts=tuple(sorted(phases.items())),
        tactical_counts=tuple(sorted(tactics.items())),
        value_bucket_counts=tuple(sorted(values.items())),
        outcome_counts=tuple(sorted(outcomes.items())),
        material_balance_counts=tuple(sorted(balances.items())),
        distinct_material_signatures=len(material_signatures),
        distinct_position_signatures=len(position_signatures),
        teacher_telemetry_samples=telemetry_count,
        teacher_telemetry_ratio=telemetry_count / sample_count,
        teacher_action_changes=action_changes,
        teacher_action_change_ratio=action_changes / telemetry_count if telemetry_count else 0.0,
        mean_teacher_policy_tv=(
            sum(record.teacher_policy_tv or 0.0 for record in telemetry) / telemetry_count
            if telemetry_count
            else 0.0
        ),
        mean_teacher_policy_kl=(
            sum(record.teacher_policy_kl or 0.0 for record in telemetry) / telemetry_count
            if telemetry_count
            else 0.0
        ),
        comparable_teacher_deltas=len(deltas),
        positive_teacher_delta_ratio=positive_delta_ratio,
        mean_teacher_delta=mean_delta,
        passed=not reasons,
        reasons=tuple(reasons),
    )
