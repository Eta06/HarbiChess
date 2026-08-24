import random

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, Side, TerminalResult
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.mcts import MCTS, SearchConfig, SearchNode


class MateEvaluator:
    def __init__(self, rules: PythonChessRules, mating_move: ChessMove) -> None:
        self.rules = rules
        self.mating_move = mating_move

    def evaluate(self, state) -> PositionEvaluation:
        return PositionEvaluation(
            tuple(
                (move, 100.0 if move == self.mating_move else 1.0)
                for move in self.rules.legal_moves(state)
            ),
            0.0,
        )


def test_terminal_results_are_antisymmetric_between_sides() -> None:
    rules = PythonChessRules()
    state = rules.initial_state()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        state = rules.apply(state, ChessMove(move))
    outcome = rules.outcome(state)

    assert outcome is not None
    assert outcome.result is TerminalResult.BLACK_WIN
    assert outcome.value_for(Side.WHITE) == -outcome.value_for(Side.BLACK)
    assert outcome.value_for(rules.view(state).side_to_move) == -1


def test_backpropagation_alternates_value_at_every_ply() -> None:
    path = [SearchNode(), SearchNode(), SearchNode(), SearchNode()]
    MCTS._backpropagate(path, -1.0)

    assert [node.value_sum for node in path] == [1.0, -1.0, 1.0, -1.0]
    assert all(node.visit_count == 1 for node in path)


def test_mate_value_is_positive_for_root_side_and_negative_after_move() -> None:
    rules = PythonChessRules()
    state = rules.initial_state("7k/8/6K1/8/8/8/8/7Q w - - 0 1")
    mating_move = ChessMove("h1h7")
    search = MCTS(
        MateEvaluator(rules, mating_move),
        rules=rules,
        config=SearchConfig(simulations=32),
    )

    result = search.search(state, rng=random.Random(5))
    terminal = search.search(rules.apply(state, mating_move), rng=random.Random(5))

    assert result.moves[0].move == mating_move
    assert result.moves[0].mean_value == 1.0
    assert terminal.root_value == -1.0
