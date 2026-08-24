import threading

import pytest

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import BackendCapabilities, PolicyValueOutput
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.selfplay.parallel import run_parallel_searches


class UniformBackend:
    capabilities = BackendCapabilities("uniform", "cpu", False, False)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.lock = threading.Lock()

    def evaluate(self, positions):
        with self.lock:
            self.batch_sizes.append(len(positions))
        output = PolicyValueOutput((0.0,) * POLICY_SIZE, (0.0, 0.0, 0.0))
        return [output] * len(positions)


def test_parallel_games_share_batches_but_keep_unique_random_streams() -> None:
    rules = PythonChessRules()
    backend = UniformBackend()
    with SharedBatchEvaluator(
        backend,
        max_batch_size=4,
        max_wait_seconds=0.02,
    ) as batches:
        mcts = MCTS(
            NeuralPositionEvaluator(batches, rules=rules),
            rules=rules,
            config=SearchConfig(simulations=4),
        )
        results = run_parallel_searches(
            mcts,
            [rules.initial_state()] * 4,
            [101, 202, 303, 404],
            max_workers=4,
            temperature=1.0,
        )

    assert [result.game_seed for result in results] == [101, 202, 303, 404]
    legal_moves = rules.legal_moves(rules.initial_state())
    assert all(result.selected_move in legal_moves for result in results)
    assert any(batch_size > 1 for batch_size in backend.batch_sizes)


def test_parallel_search_rejects_correlated_or_mismatched_configuration() -> None:
    rules = PythonChessRules()
    batches = SharedBatchEvaluator(UniformBackend(), max_batch_size=1)
    mcts = MCTS(
        NeuralPositionEvaluator(batches),
        rules=rules,
        config=SearchConfig(simulations=1),
    )
    try:
        with pytest.raises(ValueError, match="unique"):
            run_parallel_searches(
                mcts,
                [rules.initial_state()] * 2,
                [7, 7],
                max_workers=2,
                temperature=1.0,
            )
        with pytest.raises(ValueError, match="exactly one"):
            run_parallel_searches(
                mcts,
                [rules.initial_state()],
                [],
                max_workers=1,
                temperature=1.0,
            )
    finally:
        batches.close()
