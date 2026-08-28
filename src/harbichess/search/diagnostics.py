"""Fixed tactical diagnostics for search semantics and budget scaling."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import chess

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.search.evaluator import SearchEvaluator
from harbichess.search.mcts import MCTS, SearchConfig


@dataclass(frozen=True, slots=True)
class TacticalCase:
    name: str
    category: str
    fen: str
    expected_moves: tuple[str, ...]


TACTICAL_CASES = (
    TacticalCase("mate-in-one-a", "mate-in-one", "8/8/8/8/8/8/8/k1KQ4 w - - 0 1", ("d1a4",)),
    TacticalCase("mate-in-one-b", "mate-in-one", "8/8/8/8/8/8/8/k1K1Q3 w - - 0 1", ("e1a5",)),
    TacticalCase("mate-in-two-a", "mate-in-two", "8/8/8/8/3Q4/k7/8/1K6 w - - 0 1", ("b1c2",)),
    TacticalCase("mate-in-two-b", "mate-in-two", "8/8/8/8/8/k7/8/1QK5 w - - 0 1", ("b1b5",)),
    TacticalCase("forced-defense-a", "forced-defense", "8/8/8/8/8/8/k1Q5/2K5 b - - 0 1", ("a2a3",)),
    TacticalCase("forced-defense-b", "forced-defense", "8/8/8/8/8/1Q6/k7/2K5 b - - 0 1", ("a2b3",)),
    TacticalCase("hanging-queen", "hanging-piece", "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", ("e4d5",)),
    TacticalCase("hanging-rook", "hanging-piece", "4k3/8/8/3q4/3R4/8/8/4K3 b - - 0 1", ("d5d4",)),
)

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _mate_in_one_moves(board: chess.Board) -> tuple[chess.Move, ...]:
    mating = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            mating.append(move)
        board.pop()
    return tuple(mating)


def _mate_in_two_moves(board: chess.Board) -> tuple[chess.Move, ...]:
    mating = []
    for move in list(board.legal_moves):
        board.push(move)
        replies = list(board.legal_moves)
        forces_mate = not board.is_checkmate() and bool(replies)
        for reply in replies:
            board.push(reply)
            if not _mate_in_one_moves(board):
                forces_mate = False
            board.pop()
            if not forces_mate:
                break
        board.pop()
        if forces_mate:
            mating.append(move)
    return tuple(mating)


def _safe_defenses(board: chess.Board) -> tuple[chess.Move, ...]:
    safe = []
    for move in list(board.legal_moves):
        board.push(move)
        if not _mate_in_one_moves(board):
            safe.append(move)
        board.pop()
    return tuple(safe)


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    square = move.to_square
    if board.is_en_passant(move):
        square += -8 if board.turn is chess.WHITE else 8
    captured = board.piece_at(square)
    return _PIECE_VALUES[captured.piece_type] if captured is not None else 0


def validate_tactical_cases(cases: tuple[TacticalCase, ...] = TACTICAL_CASES) -> None:
    """Prove every expected move from rules instead of trusting fixture labels."""

    for case in cases:
        board = chess.Board(case.fen)
        expected = {chess.Move.from_uci(move) for move in case.expected_moves}
        if case.category == "mate-in-one":
            actual = set(_mate_in_one_moves(board))
        elif case.category == "mate-in-two":
            actual = set(_mate_in_two_moves(board))
        elif case.category == "forced-defense":
            actual = set(_safe_defenses(board))
        elif case.category == "hanging-piece":
            values = {move: _capture_value(board, move) for move in board.legal_moves}
            maximum = max(values.values())
            actual = {move for move, value in values.items() if value == maximum and value > 0}
        else:
            raise ValueError(f"unknown tactical category: {case.category}")
        if actual != expected:
            raise ValueError(
                f"invalid tactical oracle for {case.name}: expected "
                f"{sorted(move.uci() for move in expected)}, got "
                f"{sorted(move.uci() for move in actual)}"
            )


def _argmax_prior(evaluation: Any) -> ChessMove:
    return min(evaluation.priors, key=lambda item: (-item[1], item[0].uci))[0]


def run_tactical_sweep(
    evaluator: SearchEvaluator,
    *,
    rules: PythonChessRules,
    budgets: tuple[int, ...],
    workers: int,
    seed: int,
    root_fpu_reduction: float = 0.0,
    fpu_reduction: float = 0.0,
    cases: tuple[TacticalCase, ...] = TACTICAL_CASES,
) -> dict[str, Any]:
    """Measure raw and clean-search tactical choices under a fixed budget schedule."""

    if not budgets or any(budget <= 0 for budget in budgets) or workers <= 0:
        raise ValueError("tactical sweep requires positive budgets and workers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("tactical budgets must be unique and increasing")
    validate_tactical_cases(cases)
    states = tuple(rules.initial_state(case.fen) for case in cases)
    with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as pool:
        raw = tuple(pool.map(evaluator.evaluate, states))
    raw_rows = tuple(
        {
            "case": case.name,
            "category": case.category,
            "selected_move": _argmax_prior(evaluation).uci,
            "solved": _argmax_prior(evaluation).uci in case.expected_moves,
            "value": evaluation.value,
            "expected_policy_mass": sum(
                prior for move, prior in evaluation.priors if move.uci in case.expected_moves
            ),
            "top_policy_mass": max(prior for _, prior in evaluation.priors),
        }
        for case, evaluation in zip(cases, raw, strict=True)
    )
    sweeps = []
    previous_solved: set[str] = set()
    for budget in budgets:
        search = MCTS(
            evaluator,
            rules=rules,
            config=SearchConfig(
                simulations=budget,
                dirichlet_fraction=0.0,
                root_fpu_reduction=root_fpu_reduction,
                fpu_reduction=fpu_reduction,
            ),
        )

        def inspect(
            index: int,
            search: MCTS = search,
            budget: int = budget,
        ) -> dict[str, Any]:
            case = cases[index]
            result = search.search(
                states[index],
                rng=random.Random(f"{seed}:{case.name}:{budget}"),
                add_root_noise=False,
            )
            selected = result.select_move(temperature=0.0, rng=random.Random(0))
            by_move = {statistics.move.uci: statistics for statistics in result.moves}
            expected = tuple(by_move[move] for move in case.expected_moves)
            leader, runner = result.moves[:2]
            return {
                "case": case.name,
                "category": case.category,
                "selected_move": selected.uci,
                "solved": selected.uci in case.expected_moves,
                "root_value": result.root_value,
                "leader_visits": leader.visits,
                "runner_visits": runner.visits,
                "visit_margin": leader.visits - runner.visits,
                "q_margin": leader.mean_value - runner.mean_value,
                "visited_children": sum(move.visits > 0 for move in result.moves),
                "legal_children": len(result.moves),
                "expected_max_visits": max(move.visits for move in expected),
                "expected_max_value": max(move.mean_value for move in expected),
            }

        with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as pool:
            rows = tuple(pool.map(inspect, range(len(cases))))
        solved = {row["case"] for row in rows if row["solved"]}
        categories = sorted({case.category for case in cases})
        sweeps.append(
            {
                "budget": budget,
                "solved": len(solved),
                "total": len(cases),
                "solve_rate": len(solved) / len(cases),
                "regressions": sorted(previous_solved - solved),
                "category_solved": {
                    category: sum(
                        row["solved"] for row in rows if row["category"] == category
                    )
                    for category in categories
                },
                "cases": rows,
            }
        )
        previous_solved = solved
    counts = tuple(sweep["solved"] for sweep in sweeps)
    return {
        "cases": tuple(asdict(case) for case in cases),
        "raw": {
            "solved": sum(row["solved"] for row in raw_rows),
            "total": len(cases),
            "cases": raw_rows,
        },
        "budgets": sweeps,
        "aggregate_solve_count_monotonic": all(
            current >= previous for previous, current in pairwise(counts)
        ),
    }
