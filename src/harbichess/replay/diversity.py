"""Self-play diversity and coverage measurements for collapse detection."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import TerminalResult
from harbichess.selfplay.game import SelfPlayGame


@dataclass(frozen=True, slots=True)
class OpeningCoverage:
    ply: int
    eligible_games: int
    unique_prefixes: int
    entropy: float
    effective_prefixes: float


@dataclass(frozen=True, slots=True)
class TerminationCoverage:
    termination: str
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class DiversityMetrics:
    games: int
    positions: int
    unique_game_ratio: float
    duplicate_game_ratio: float
    unique_position_ratio: float
    selected_actions: int
    action_space_coverage: float
    mean_policy_entropy: float
    effective_policy_branches: float
    mean_game_plies: float
    white_wins: int
    draws: int
    black_wins: int
    decisive_games: int
    decisive_game_ratio: float
    max_ply_draws: int
    max_ply_draw_ratio: float
    repetition_redirects: int
    repetition_redirect_ratio: float
    root_search_evaluated: int
    root_search_adjustments: int
    root_search_adjustment_ratio: float
    mean_adjusted_root_margin: float
    terminations: tuple[TerminationCoverage, ...]
    openings: tuple[OpeningCoverage, ...]


def _entropy(counts: Counter[object]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def measure_diversity(
    games: Sequence[SelfPlayGame],
    *,
    rules: PythonChessRules | None = None,
    opening_plies: tuple[int, ...] = (4, 8, 12),
) -> DiversityMetrics:
    if not games or any(ply <= 0 for ply in opening_plies):
        raise ValueError("diversity requires games and positive opening depths")
    engine = rules or PythonChessRules()
    game_lines = [tuple(move.uci for move in game.final_state.moves) for game in games]
    unique_games = len(set(game_lines))
    position_keys: set[str] = set()
    selected_actions: set[int] = set()
    policy_entropies = []
    position_count = 0
    for game in games:
        for sample in game.samples:
            board = engine.board(sample.state)
            position_keys.add(" ".join(board.fen().split()[:4]))
            selected_actions.add(move_to_action(board, board.parse_uci(sample.selected_move.uci)))
            policy_entropies.append(
                -sum(
                    probability * math.log(probability)
                    for _, probability in sample.visit_policy
                    if probability > 0
                )
            )
            position_count += 1

    openings = []
    for ply in opening_plies:
        prefixes = Counter(line[:ply] for line in game_lines if len(line) >= ply)
        entropy = _entropy(prefixes)
        openings.append(
            OpeningCoverage(
                ply=ply,
                eligible_games=sum(prefixes.values()),
                unique_prefixes=len(prefixes),
                entropy=entropy,
                effective_prefixes=math.exp(entropy),
            )
        )

    outcomes = Counter(game.outcome.result for game in games)
    terminations = Counter(game.outcome.termination for game in games)
    decisive_games = outcomes[TerminalResult.WHITE_WIN] + outcomes[TerminalResult.BLACK_WIN]
    max_ply_draws = sum(game.outcome.termination == "max_plies" for game in games)
    mean_policy_entropy = sum(policy_entropies) / len(policy_entropies) if policy_entropies else 0.0
    repetition_redirects = sum(
        sample.repetition_redirected for game in games for sample in game.samples
    )
    root_evidence = [
        sample
        for game in games
        for sample in game.samples
        if sample.root_search_final_margin is not None
    ]
    root_adjustments = [sample for sample in root_evidence if sample.root_search_adjusted]
    return DiversityMetrics(
        games=len(games),
        positions=position_count,
        unique_game_ratio=unique_games / len(games),
        duplicate_game_ratio=1.0 - unique_games / len(games),
        unique_position_ratio=len(position_keys) / position_count if position_count else 0.0,
        selected_actions=len(selected_actions),
        action_space_coverage=len(selected_actions) / POLICY_SIZE,
        mean_policy_entropy=mean_policy_entropy,
        effective_policy_branches=math.exp(mean_policy_entropy),
        mean_game_plies=sum(game.final_state.ply for game in games) / len(games),
        white_wins=outcomes[TerminalResult.WHITE_WIN],
        draws=outcomes[TerminalResult.DRAW],
        black_wins=outcomes[TerminalResult.BLACK_WIN],
        decisive_games=decisive_games,
        decisive_game_ratio=decisive_games / len(games),
        max_ply_draws=max_ply_draws,
        max_ply_draw_ratio=max_ply_draws / len(games),
        repetition_redirects=repetition_redirects,
        repetition_redirect_ratio=(
            repetition_redirects / position_count if position_count else 0.0
        ),
        root_search_evaluated=len(root_evidence),
        root_search_adjustments=len(root_adjustments),
        root_search_adjustment_ratio=(
            len(root_adjustments) / len(root_evidence) if root_evidence else 0.0
        ),
        mean_adjusted_root_margin=(
            sum(sample.root_search_final_margin or 0.0 for sample in root_adjustments)
            / len(root_adjustments)
            if root_adjustments
            else 0.0
        ),
        terminations=tuple(
            TerminationCoverage(termination, count, count / len(games))
            for termination, count in sorted(terminations.items())
        ),
        openings=tuple(openings),
    )
