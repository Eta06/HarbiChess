import math

import chess
import pytest

from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition, PolicyValueOutput
from harbichess.core.state import ChessMove
from harbichess.search.evaluator import NeuralPositionEvaluator


class FixedEvaluator:
    def __init__(self, output: PolicyValueOutput) -> None:
        self.output = output

    def evaluate(self, position: EncodedPosition) -> PolicyValueOutput:
        assert position.shape == (8, 8, 104)
        return self.output


def test_neural_evaluator_masks_policy_and_converts_wdl_value() -> None:
    board = chess.Board()
    logits = [0.0] * POLICY_SIZE
    logits[move_to_action(board, chess.Move.from_uci("e2e4"))] = math.log(4)
    output = PolicyValueOutput(tuple(logits), (math.log(4), math.log(2), 0.0))
    rules = PythonChessRules()
    evaluator = NeuralPositionEvaluator(FixedEvaluator(output), rules=rules)

    result = evaluator.evaluate(rules.initial_state())

    assert len(result.priors) == 20
    assert sum(prior for _, prior in result.priors) == pytest.approx(1.0)
    assert max(result.priors, key=lambda item: item[1])[0] == ChessMove("e2e4")
    assert result.value == pytest.approx(3 / 7)


def test_neural_evaluator_rejects_terminal_and_wrong_policy_shape() -> None:
    rules = PythonChessRules()
    terminal = rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    evaluator = NeuralPositionEvaluator(
        FixedEvaluator(PolicyValueOutput((0.0,), (0.0, 0.0, 0.0))),
        rules=rules,
    )
    with pytest.raises(ValueError, match="terminal"):
        evaluator.evaluate(terminal)
    with pytest.raises(ValueError, match="4672"):
        evaluator.evaluate(rules.initial_state())
