"""Shared blocking-to-batched inference bridge for parallel searches."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from harbichess.core.backend import (
    EncodedPosition,
    MaskedPolicyValueOutput,
    PolicyValueBackend,
    PolicyValueOutput,
)


@dataclass(slots=True)
class _Request:
    position: EncodedPosition
    action_indices: tuple[int, ...] | None
    future: Future[PolicyValueOutput | MaskedPolicyValueOutput]
    queued_at: float


@dataclass(frozen=True, slots=True)
class BatchStatistics:
    batches: int
    positions: int
    largest_batch: int
    backend_seconds: float
    queue_wait_seconds: float

    @property
    def average_batch_size(self) -> float:
        return self.positions / self.batches if self.batches else 0.0

    @property
    def average_queue_wait_ms(self) -> float:
        return self.queue_wait_seconds / self.positions * 1_000 if self.positions else 0.0


class SharedBatchEvaluator:
    """Coalesce synchronous search requests onto one backend worker thread."""

    def __init__(
        self,
        backend: PolicyValueBackend,
        *,
        max_batch_size: int = 128,
        max_wait_seconds: float = 0.001,
    ) -> None:
        if max_batch_size <= 0 or max_wait_seconds < 0:
            raise ValueError("batch size must be positive and wait time non-negative")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._batches = 0
        self._positions = 0
        self._largest_batch = 0
        self._backend_seconds = 0.0
        self._queue_wait_seconds = 0.0
        self._worker = threading.Thread(
            target=self._run,
            name="harbichess-inference",
            daemon=True,
        )
        self._worker.start()

    def evaluate(self, position: EncodedPosition) -> PolicyValueOutput:
        future: Future[PolicyValueOutput] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("shared evaluator is closed")
            self._queue.put(_Request(position, None, future, time.perf_counter()))
        return future.result()

    def evaluate_masked(
        self,
        position: EncodedPosition,
        action_indices: tuple[int, ...],
    ) -> MaskedPolicyValueOutput:
        if not action_indices:
            raise ValueError("masked evaluation requires legal action indices")
        future: Future[PolicyValueOutput | MaskedPolicyValueOutput] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("shared evaluator is closed")
            self._queue.put(_Request(position, action_indices, future, time.perf_counter()))
        output = future.result()
        if not isinstance(output, MaskedPolicyValueOutput):
            raise TypeError("masked backend returned a full policy output")
        return output

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._worker.join()

    @property
    def statistics(self) -> BatchStatistics:
        with self._lock:
            return BatchStatistics(
                self._batches,
                self._positions,
                self._largest_batch,
                self._backend_seconds,
                self._queue_wait_seconds,
            )

    def reset_statistics(self) -> BatchStatistics:
        """Return and clear counters after callers have drained their requests."""
        with self._lock:
            statistics = BatchStatistics(
                self._batches,
                self._positions,
                self._largest_batch,
                self._backend_seconds,
                self._queue_wait_seconds,
            )
            self._batches = 0
            self._positions = 0
            self._largest_batch = 0
            self._backend_seconds = 0.0
            self._queue_wait_seconds = 0.0
            return statistics

    def __enter__(self) -> SharedBatchEvaluator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            requests = [first]
            stop_after_batch = False
            deadline = time.perf_counter() + self.max_wait_seconds
            while len(requests) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    stop_after_batch = True
                    break
                requests.append(request)
            full = [request for request in requests if request.action_indices is None]
            masked = [request for request in requests if request.action_indices is not None]
            if full:
                self._evaluate_batch(full)
            if masked:
                self._evaluate_batch(masked)
            if stop_after_batch:
                return

    def _evaluate_batch(self, requests: list[_Request]) -> None:
        started = time.perf_counter()
        with self._lock:
            self._batches += 1
            self._positions += len(requests)
            self._largest_batch = max(self._largest_batch, len(requests))
            self._queue_wait_seconds += sum(started - request.queued_at for request in requests)
        try:
            positions = [request.position for request in requests]
            if requests[0].action_indices is None:
                outputs = self.backend.evaluate(positions)
            else:
                action_indices = [
                    request.action_indices
                    for request in requests
                    if request.action_indices is not None
                ]
                evaluate_masked = getattr(self.backend, "evaluate_masked", None)
                if evaluate_masked is not None:
                    outputs = evaluate_masked(positions, action_indices)
                else:
                    full_outputs = self.backend.evaluate(positions)
                    outputs = [
                        MaskedPolicyValueOutput(
                            tuple(output.policy_logits[action] for action in indices),
                            output.wdl_logits,
                        )
                        for output, indices in zip(
                            full_outputs,
                            action_indices,
                            strict=True,
                        )
                    ]
            if len(outputs) != len(requests):
                raise RuntimeError(
                    f"backend returned {len(outputs)} outputs for {len(requests)} positions"
                )
        except Exception as error:
            for request in requests:
                request.future.set_exception(error)
            return
        backend_seconds = time.perf_counter() - started
        with self._lock:
            self._backend_seconds += backend_seconds
        for request, output in zip(requests, outputs, strict=True):
            request.future.set_result(output)
