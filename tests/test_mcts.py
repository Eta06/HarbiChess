import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, TerminalResult
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.mcts import MCTS, SearchConfig


class UniformEvaluator:
    def __init__(self, rules: PythonChessRules, preferred: ChessMove | None = None) -> None:
        self.rules = rules
        self.preferred = preferred

    def evaluate(self, state) -> PositionEvaluation:
        moves = self.rules.legal_moves(state)
        priors = tuple(
            (move, 20.0 if move == self.preferred else 1.0)
            for move in moves
        )
        return PositionEvaluation(priors, 0.0)


def test_mcts_accounts_for_every_simulation_and_returns_legal_move() -> None:
    rules = PythonChessRules()
    search = MCTS(
        UniformEvaluator(rules),
        rules=rules,
        config=SearchConfig(simulations=40),
    )

    result = search.search(rules.initial_state(), rng=random.Random(7))

    assert sum(move.visits for move in result.moves) == 40
    assert len(result.moves) == 20
    selected = result.select_move(temperature=0, rng=random.Random(7))
    assert selected in rules.legal_moves(rules.initial_state())


def test_mcts_discovers_preferred_mate_in_one() -> None:
    rules = PythonChessRules()
    state = rules.initial_state("7k/8/6K1/8/8/8/8/7Q w - - 0 1")
    mating_move = ChessMove("h1h7")
    search = MCTS(
        UniformEvaluator(rules, mating_move),
        rules=rules,
        config=SearchConfig(simulations=48),
    )

    result = search.search(state, rng=random.Random(11))

    assert result.select_move(temperature=0, rng=random.Random(11)) == mating_move
    assert result.moves[0].mean_value > 0


def test_terminal_search_and_seeded_root_noise_are_deterministic() -> None:
    rules = PythonChessRules()
    terminal = rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    search = MCTS(
        UniformEvaluator(rules),
        rules=rules,
        config=SearchConfig(simulations=12),
    )
    terminal_result = search.search(terminal, rng=random.Random(1))
    first = search.search(rules.initial_state(), rng=random.Random(3), add_root_noise=True)
    second = search.search(rules.initial_state(), rng=random.Random(3), add_root_noise=True)

    assert terminal_result.outcome is not None
    assert terminal_result.outcome.result is TerminalResult.DRAW
    assert terminal_result.moves == ()
    assert first.moves == second.moves
    with pytest.raises(ValueError, match="terminal"):
        terminal_result.select_move(temperature=0, rng=random.Random(1))

    checkmated = rules.initial_state()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        checkmated = rules.apply(checkmated, ChessMove(move))
    assert search.search(checkmated, rng=random.Random(1)).root_value == -1.0


def test_search_config_and_temperature_validation() -> None:
    with pytest.raises(ValueError, match="simulations"):
        SearchConfig(simulations=0)
    result = MCTS(
        UniformEvaluator(PythonChessRules()),
        config=SearchConfig(simulations=1),
    ).search(PythonChessRules().initial_state(), rng=random.Random(1))
    with pytest.raises(ValueError, match="temperature"):
        result.select_move(temperature=-1, rng=random.Random(1))
    assert result.select_move(temperature=0.001, rng=random.Random(1)) in {
        move.move for move in result.moves
    }
