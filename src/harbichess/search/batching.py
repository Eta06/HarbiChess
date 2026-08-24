"""Shared blocking-to-batched inference bridge for parallel searches."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from harbichess.core.backend import EncodedPosition, PolicyValueBackend, PolicyValueOutput


@dataclass(slots=True)
class _Request:
    position: EncodedPosition
    future: Future[PolicyValueOutput]


@dataclass(frozen=True, slots=True)
class BatchStatistics:
    batches: int
    positions: int
    largest_batch: int

    @property
    def average_batch_size(self) -> float:
        return self.positions / self.batches if self.batches else 0.0


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
            self._queue.put(_Request(position, future))
        return future.result()

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
            return BatchStatistics(self._batches, self._positions, self._largest_batch)

    def reset_statistics(self) -> BatchStatistics:
        """Return and clear counters after callers have drained their requests."""
        with self._lock:
            statistics = BatchStatistics(self._batches, self._positions, self._largest_batch)
            self._batches = 0
            self._positions = 0
            self._largest_batch = 0
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
            self._evaluate_batch(requests)
            if stop_after_batch:
                return

    def _evaluate_batch(self, requests: list[_Request]) -> None:
        with self._lock:
            self._batches += 1
            self._positions += len(requests)
            self._largest_batch = max(self._largest_batch, len(requests))
        try:
            outputs = self.backend.evaluate([request.position for request in requests])
            if len(outputs) != len(requests):
                raise RuntimeError(
                    f"backend returned {len(outputs)} outputs for {len(requests)} positions"
                )
        except Exception as error:
            for request in requests:
                request.future.set_exception(error)
            return
        for request, output in zip(requests, outputs, strict=True):
            request.future.set_result(output)
