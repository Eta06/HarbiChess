import threading

import pytest

from harbichess.core.backend import (
    BackendCapabilities,
    EncodedPosition,
    MaskedPolicyValueOutput,
    PolicyValueOutput,
)
from harbichess.search.batching import SharedBatchEvaluator


class RecordingBackend:
    capabilities = BackendCapabilities("recording", "cpu", False, False)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate(self, positions: list[EncodedPosition]) -> list[PolicyValueOutput]:
        self.batch_sizes.append(len(positions))
        return [PolicyValueOutput(position.values, (0.0, 0.0, 0.0)) for position in positions]


def test_parallel_requests_share_one_backend_batch() -> None:
    backend = RecordingBackend()
    barrier = threading.Barrier(9)
    outputs: list[PolicyValueOutput] = []
    with SharedBatchEvaluator(backend, max_batch_size=8, max_wait_seconds=0.05) as evaluator:
        def evaluate(index: int) -> None:
            barrier.wait()
            outputs.append(evaluator.evaluate(EncodedPosition((float(index),), (1,), 1)))

        threads = [threading.Thread(target=evaluate, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

    assert backend.batch_sizes == [8]
    assert sorted(output.policy_logits[0] for output in outputs) == list(map(float, range(8)))
    assert evaluator.statistics.batches == 1
    assert evaluator.statistics.positions == 8
    assert evaluator.statistics.largest_batch == 8
    assert evaluator.statistics.average_batch_size == 8.0


def test_evaluator_rejects_use_after_close_and_invalid_config() -> None:
    evaluator = SharedBatchEvaluator(RecordingBackend(), max_batch_size=1)
    evaluator.close()
    evaluator.close()
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate(EncodedPosition((0.0,), (1,), 1))
    with pytest.raises(ValueError, match="batch size"):
        SharedBatchEvaluator(RecordingBackend(), max_batch_size=0)


def test_masked_requests_fall_back_to_generic_backend_slicing() -> None:
    backend = RecordingBackend()
    with SharedBatchEvaluator(backend, max_batch_size=1) as evaluator:
        output = evaluator.evaluate_masked(
            EncodedPosition((1.0, 2.0, 3.0), (3,), 1),
            (2, 0),
        )

    assert isinstance(output, MaskedPolicyValueOutput)
    assert output.policy_logits == (3.0, 1.0)
