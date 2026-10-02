import math
import random

import chess
import pytest
import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
from harbichess.search.mcts import MCTS, SearchConfig


@pytest.mark.parametrize("black", [False, True])
@pytest.mark.parametrize("algorithm", ["gumbel", "puct"])
def test_real_backend_search_terminal_sign_and_policy_semantics(black, algorithm):
    torch.set_num_threads(1)
    rules = PythonChessRules()
    board = chess.Board("8/8/8/8/8/8/8/k1KQ4 w - - 0 1")
    if black:
        board = board.mirror()
    state = rules.initial_state(board.fen())
    network = TorchChessNetwork(NetworkConfig(trunk_channels=4, residual_blocks=1))
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.value_output.bias.copy_(torch.tensor([0.2, 0.3, -0.5]))
    bridge = SharedBatchEvaluator(TorchPolicyValueBackend(network), max_wait_seconds=0)
    try:
        evaluator = NeuralPositionEvaluator(bridge, rules=rules)
        evaluation = evaluator.evaluate(state)
        denominator = sum(math.exp(x) for x in (0.2, 0.3, -0.5))
        assert evaluation.value == pytest.approx((math.exp(0.2) - math.exp(-0.5)) / denominator)
        assert sum(p for _, p in evaluation.priors) == pytest.approx(1)
        assert all(m in rules.legal_moves(state) for m, _ in evaluation.priors)
        if algorithm == "gumbel":
            search = FullGumbelMCTS(
                evaluator,
                rules=rules,
                config=FullGumbelConfig(simulations=64, max_considered_actions=32),
            )
        else:
            search = MCTS(evaluator, rules=rules, config=SearchConfig(simulations=64))
        result = search.search(state, rng=random.Random(9))
        mating = [
            m
            for m in result.moves
            if (outcome := rules.outcome(rules.apply(state, m.move))) is not None
            and outcome.value_for(rules.view(state).side_to_move) == 1
        ]
        assert mating and max(m.mean_value for m in mating) == 1.0
        terminal = search.search(rules.apply(state, mating[0].move), rng=random.Random(9))
        assert terminal.root_value == -1.0
    finally:
        bridge.close()
