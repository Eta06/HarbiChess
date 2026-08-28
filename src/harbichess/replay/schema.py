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
TARGET_SCHEMA_VERSION = 12
SUPPORTED_TARGET_SCHEMA_VERSIONS = frozenset(
    {3, 4, 5, 6, 7, 8, 9, 10, 11, TARGET_SCHEMA_VERSION}
)


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
class RepetitionRiskEstimate:
    action: int
    horizon_plies: int
    rollouts: int
    repetition_events: int
    estimated_risk: float
    upper_confidence_bound: float
    loop_value_samples: int = 0
    exact_loop_value_samples: int = 0
    mean_loop_value: float | None = None
    lower_loop_value_bound: float | None = None
    risk_adjusted_value_lower_bound: float | None = None

    def __post_init__(self) -> None:
        if (
            not 0 <= self.action < POLICY_SIZE
            or self.horizon_plies not in (2, 3)
            or self.rollouts <= 0
            or not 0 <= self.repetition_events <= self.rollouts
        ):
            raise ValueError("repetition risk identity and counts are invalid")
        expected = self.repetition_events / self.rollouts
        if not math.isclose(self.estimated_risk, expected, abs_tol=1e-12):
            raise ValueError("estimated repetition risk must match its event count")
        if not (
            math.isfinite(self.upper_confidence_bound)
            and 0.0 <= self.estimated_risk <= self.upper_confidence_bound <= 1.0
        ):
            raise ValueError("repetition risk confidence bound must contain the estimate")
        loop_values = (self.mean_loop_value, self.lower_loop_value_bound)
        if self.loop_value_samples:
            if not 0 < self.loop_value_samples <= self.repetition_events:
                raise ValueError("loop value samples must correspond to repetition events")
            if any(value is None or not math.isfinite(value) for value in loop_values):
                raise ValueError("loop value estimates must be finite when sampled")
            assert self.mean_loop_value is not None
            assert self.lower_loop_value_bound is not None
            if not -1.0 <= self.lower_loop_value_bound <= self.mean_loop_value <= 1.0:
                raise ValueError("loop value lower bound must contain the mean")
        elif any(value is not None for value in loop_values):
            raise ValueError("loop values require at least one sampled repetition event")
        if not 0 <= self.exact_loop_value_samples <= self.loop_value_samples:
            raise ValueError("exact loop values must be a subset of loop value samples")
        if self.risk_adjusted_value_lower_bound is not None and not (
            math.isfinite(self.risk_adjusted_value_lower_bound)
            and -1.0 <= self.risk_adjusted_value_lower_bound <= 1.0
        ):
            raise ValueError("risk-adjusted branch value must be finite and bounded")


@dataclass(frozen=True, slots=True)
class ContinuationEvidence:
    method_version: int
    confidence_level: float
    branch_searches: int
    simulations_per_search: int
    repeat_value: float
    minimum_advantage: float
    repeat_actions: tuple[int, ...]
    branches: tuple[BranchValueEstimate, ...]
    qualified_actions: tuple[int, ...]
    source_model_sha256: str
    repetition_risks: tuple[RepetitionRiskEstimate, ...] = ()
    maximum_repetition_risk: float | None = None
    evaluated_root_value: float | None = None
    minimum_advantaged_root_value: float | None = None

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
        if not math.isfinite(self.minimum_advantage) or not 0.0 <= self.minimum_advantage <= 1.0:
            raise ValueError("continuation minimum advantage must be finite and bounded")
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
        branches_by_action = {branch.action: branch for branch in self.branches}
        if any(
            branches_by_action[action].lower_confidence_bound
            <= self.repeat_value + self.minimum_advantage
            for action in self.qualified_actions
        ):
            raise ValueError("qualified branch lower bound must clear the repeat gate")
        if len(self.source_model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_model_sha256.lower()
        ):
            raise ValueError("continuation evidence model hash must be SHA-256")
        if self.repetition_risks:
            risk_actions = {risk.action for risk in self.repetition_risks}
            if (
                len(risk_actions) != len(self.repetition_risks)
                or not risk_actions <= branch_actions
            ):
                raise ValueError("repetition risk actions must be unique evaluated branches")
            risks_by_action = {risk.action: risk for risk in self.repetition_risks}
            if self.method_version == 2:
                if not (
                    self.maximum_repetition_risk is not None
                    and math.isfinite(self.maximum_repetition_risk)
                    and 0.0 <= self.maximum_repetition_risk <= 1.0
                ):
                    raise ValueError("repetition risk evidence requires a bounded maximum")
                if any(
                    action not in risks_by_action
                    or risks_by_action[action].upper_confidence_bound
                    > self.maximum_repetition_risk
                    for action in self.qualified_actions
                ):
                    raise ValueError("qualified branch must clear the repetition risk gate")
            elif self.method_version >= 3:
                if self.maximum_repetition_risk is not None:
                    raise ValueError("value-aware evidence must not use a probability cutoff")
                if not (
                    self.evaluated_root_value is not None
                    and self.minimum_advantaged_root_value is not None
                    and math.isfinite(self.evaluated_root_value)
                    and math.isfinite(self.minimum_advantaged_root_value)
                    and -1.0 <= self.evaluated_root_value <= 1.0
                    and 0.0 <= self.minimum_advantaged_root_value <= 1.0
                    and self.evaluated_root_value > self.minimum_advantaged_root_value
                ):
                    raise ValueError("value-aware evidence requires an advantaged root")
                if any(
                    action not in risks_by_action
                    or risks_by_action[action].risk_adjusted_value_lower_bound is None
                    or risks_by_action[action].risk_adjusted_value_lower_bound
                    <= self.repeat_value + self.minimum_advantage
                    for action in self.qualified_actions
                ):
                    raise ValueError("qualified branch must clear the value-aware risk gate")
        elif self.maximum_repetition_risk is not None:
            raise ValueError("maximum repetition risk requires branch risk evidence")
        if self.method_version < 3 and (
            self.evaluated_root_value is not None
            or self.minimum_advantaged_root_value is not None
        ):
            raise ValueError("legacy continuation evidence cannot contain root advantage fields")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContinuationEvidence:
        parsed = dict(data)
        parsed["repeat_actions"] = tuple(int(action) for action in parsed["repeat_actions"])
        parsed["branches"] = tuple(BranchValueEstimate(**branch) for branch in parsed["branches"])
        parsed["qualified_actions"] = tuple(int(action) for action in parsed["qualified_actions"])
        parsed["repetition_risks"] = tuple(
            RepetitionRiskEstimate(**risk) for risk in parsed.get("repetition_risks", ())
        )
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class PolicyRegretAdjustment:
    method_version: int
    temperature: float
    root_value: float
    repeat_value: float
    best_nonrepeat_value: float
    regret: float
    redirect_fraction: float
    repeat_actions: tuple[int, ...]
    redirect_actions: tuple[int, ...]
    source_model_sha256: str

    def __post_init__(self) -> None:
        values = (
            self.temperature,
            self.root_value,
            self.repeat_value,
            self.best_nonrepeat_value,
            self.regret,
            self.redirect_fraction,
        )
        if self.method_version <= 0 or any(not math.isfinite(value) for value in values):
            raise ValueError("policy regret values must be finite and versioned")
        if self.temperature <= 0.0:
            raise ValueError("policy regret temperature must be positive")
        if not (
            -1.0 <= self.root_value <= 1.0
            and -1.0 <= self.repeat_value <= 1.0
            and -1.0 <= self.best_nonrepeat_value <= 1.0
            and 0.0 <= self.regret <= 2.0
            and 0.0 <= self.redirect_fraction < 1.0
        ):
            raise ValueError("policy regret values must be bounded")
        expected_regret = max(
            0.0,
            min(self.root_value, self.best_nonrepeat_value) - self.repeat_value,
        )
        if not math.isclose(self.regret, expected_regret, abs_tol=1e-12):
            raise ValueError("policy regret must match the conservative value gap")
        expected_fraction = 1.0 - math.exp(-self.regret / self.temperature)
        if not math.isclose(self.redirect_fraction, expected_fraction, abs_tol=1e-12):
            raise ValueError("redirect fraction must follow the frozen regret transform")
        if (
            not self.repeat_actions
            or len(self.repeat_actions) != len(set(self.repeat_actions))
            or not self.redirect_actions
            or len(self.redirect_actions) != len(set(self.redirect_actions))
            or set(self.repeat_actions) & set(self.redirect_actions)
        ):
            raise ValueError("policy regret actions must be unique repeat/non-repeat sets")
        if len(self.source_model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_model_sha256.lower()
        ):
            raise ValueError("policy regret model hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyRegretAdjustment:
        parsed = dict(data)
        parsed["repeat_actions"] = tuple(int(action) for action in parsed["repeat_actions"])
        parsed["redirect_actions"] = tuple(
            int(action) for action in parsed["redirect_actions"]
        )
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
    outcome_value: int | None
    repetition_redirected: bool
    continuation_evidence: ContinuationEvidence | None = None
    policy_regret_adjustment: PolicyRegretAdjustment | None = None
    root_search_adjusted: bool = False
    root_search_first_margin: float | None = None
    root_search_final_margin: float | None = None
    raw_policy: tuple[tuple[int, float], ...] = ()
    teacher_policy_tv: float | None = None
    teacher_policy_kl: float | None = None
    teacher_argmax_changed: bool | None = None
    teacher_search_value_delta: float | None = None
    behavior_target_decoupled: bool = False

    def __post_init__(self) -> None:
        if not self.game_id or self.game_index < 0 or self.seed < 0 or self.ply != len(self.moves):
            raise ValueError("replay identity and ply history are inconsistent")
        if not math.isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError("root value must be finite and between -1 and 1")
        if self.outcome_value is not None and self.outcome_value not in (-1, 0, 1):
            raise ValueError("outcome value must be -1, 0, 1, or unknown")
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
        if not self.behavior_target_decoupled and not any(
            action == self.selected_action and probability > 0
            for action, probability in self.policy
        ):
            raise ValueError("selected action must have positive replay-policy mass")
        if self.continuation_evidence is not None:
            qualified = set(self.continuation_evidence.qualified_actions)
            if not qualified or set(indices) != qualified or self.selected_action not in qualified:
                raise ValueError("confidence-gated policy must exactly match qualified actions")
        if self.policy_regret_adjustment is not None:
            adjustment = self.policy_regret_adjustment
            if (
                adjustment.redirect_fraction > 0.0
                and not set(adjustment.redirect_actions) <= set(indices)
            ):
                raise ValueError("regret redirect actions must remain in the blended policy")
            if adjustment.redirect_fraction > 0.0 and not self.repetition_redirected:
                raise ValueError("positive regret adjustment must mark the target redirected")
        root_margins = (self.root_search_first_margin, self.root_search_final_margin)
        if self.root_search_adjusted and any(margin is None for margin in root_margins):
            raise ValueError("adjusted root search requires both confidence margins")
        if any(
            margin is not None and (not math.isfinite(margin) or not -2.0 <= margin <= 2.0)
            for margin in root_margins
        ):
            raise ValueError("root-search confidence margins must be finite and bounded")
        raw_indices = [action for action, _ in self.raw_policy]
        raw_probabilities = [probability for _, probability in self.raw_policy]
        if self.raw_policy and (
            len(raw_indices) != len(set(raw_indices))
            or any(not 0 <= action < POLICY_SIZE for action in raw_indices)
            or any(not math.isfinite(value) or value < 0 for value in raw_probabilities)
            or not math.isclose(sum(raw_probabilities), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("raw network policy must be unique, legal-sized, and normalized")
        teacher_metrics = (self.teacher_policy_tv, self.teacher_policy_kl)
        if self.raw_policy:
            if any(
                value is None or not math.isfinite(value) or value < 0
                for value in teacher_metrics
            ):
                raise ValueError("raw policy requires finite non-negative teacher metrics")
            assert self.teacher_policy_tv is not None
            if self.teacher_policy_tv > 1.0 or self.teacher_argmax_changed is None:
                raise ValueError("teacher policy evidence is outside its valid range")
        elif any(value is not None for value in (*teacher_metrics, self.teacher_argmax_changed)):
            raise ValueError("teacher policy evidence requires a raw network policy")
        if self.teacher_search_value_delta is not None and (
            not math.isfinite(self.teacher_search_value_delta)
            or not -2.0 <= self.teacher_search_value_delta <= 2.0
        ):
            raise ValueError("teacher search value delta must be finite and bounded")

    @property
    def state(self) -> ChessState:
        return ChessState(self.root_fen, tuple(ChessMove(move) for move in self.moves))

    def validate_rules(self, rules: PythonChessRules) -> None:
        board = rules.board(self.state)
        expected_side = Side.WHITE if board.turn else Side.BLACK
        if self.side_to_move is not expected_side:
            raise ValueError("replay side-to-move does not match reconstructed state")
        legal_actions = {move_to_action(board, move) for move in board.legal_moves}
        if self.selected_action not in legal_actions:
            raise ValueError("selected replay action is illegal")
        if any(action not in legal_actions for action, _ in self.policy):
            raise ValueError("replay policy contains an illegal action")
        if any(action not in legal_actions for action, _ in self.raw_policy):
            raise ValueError("raw network policy contains an illegal action")
        if self.continuation_evidence is not None:
            evidence_actions = {
                *self.continuation_evidence.repeat_actions,
                *(branch.action for branch in self.continuation_evidence.branches),
            }
            if not evidence_actions <= legal_actions:
                raise ValueError("continuation evidence contains an illegal action")
        if self.policy_regret_adjustment is not None:
            adjustment_actions = {
                *self.policy_regret_adjustment.repeat_actions,
                *self.policy_regret_adjustment.redirect_actions,
            }
            if not adjustment_actions <= legal_actions:
                raise ValueError("policy regret adjustment contains an illegal action")

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
        parsed["raw_policy"] = tuple(
            (int(action), float(probability))
            for action, probability in parsed.get("raw_policy", ())
        )
        evidence = parsed.get("continuation_evidence")
        parsed["continuation_evidence"] = (
            ContinuationEvidence.from_dict(evidence) if evidence is not None else None
        )
        adjustment = parsed.get("policy_regret_adjustment")
        parsed["policy_regret_adjustment"] = (
            PolicyRegretAdjustment.from_dict(adjustment) if adjustment is not None else None
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
    raw_policy = tuple(
        sorted(
            (move_to_action(board, board.parse_uci(move.uci)), probability)
            for move, probability in sample.raw_policy
        )
    )
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
        root_search_adjusted=sample.root_search_adjusted,
        root_search_first_margin=sample.root_search_first_margin,
        root_search_final_margin=sample.root_search_final_margin,
        raw_policy=raw_policy,
        teacher_policy_tv=sample.teacher_policy_tv,
        teacher_policy_kl=sample.teacher_policy_kl,
        teacher_argmax_changed=sample.teacher_argmax_changed,
        teacher_search_value_delta=sample.teacher_search_value_delta,
        behavior_target_decoupled=sample.behavior_target_decoupled,
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
