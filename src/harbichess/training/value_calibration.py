"""Deterministic scalar calibration for production WDL logits."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class ScalarCalibration:
    """A positive multiplier selected by weighted WDL cross entropy."""

    logit_scale: float
    temperature: float
    fit_cross_entropy_before: float
    fit_cross_entropy_after: float
    iterations: int
    rows: int
    groups: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _rows(logits: mx.array) -> tuple[tuple[float, float, float], ...]:
    if logits.ndim != 2 or logits.shape[1] != 3 or logits.shape[0] == 0:
        raise ValueError("WDL logits must have shape (non-zero rows, 3)")
    mx.eval(logits)
    rows = tuple(tuple(map(float, row)) for row in logits.tolist())
    if any(not all(math.isfinite(value) for value in row) for row in rows):
        raise ValueError("WDL logits must be finite")
    return rows  # type: ignore[return-value]


def _weights(
    count: int, group_ids: Sequence[Hashable] | None
) -> tuple[tuple[float, ...], int]:
    if group_ids is None:
        return (tuple(1.0 / count for _ in range(count)), count)
    if len(group_ids) != count:
        raise ValueError("calibration group count must match logits")
    counts = Counter(group_ids)
    groups = len(counts)
    return (
        tuple(1.0 / (groups * counts[group_id]) for group_id in group_ids),
        groups,
    )


def _objective(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    weights: Sequence[float],
    scale: float,
) -> tuple[float, float]:
    loss = derivative = 0.0
    for row, label, weight in zip(rows, labels, weights, strict=True):
        scaled = tuple(scale * value for value in row)
        maximum = max(scaled)
        exponentials = tuple(math.exp(value - maximum) for value in scaled)
        total = sum(exponentials)
        probabilities = tuple(value / total for value in exponentials)
        loss += weight * (
            math.log(total) + maximum - scaled[label]
        )
        derivative += weight * (
            sum(
                probability * value
                for probability, value in zip(probabilities, row, strict=True)
            )
            - row[label]
        )
    return loss, derivative


def fit_scalar_calibration(
    logits: mx.array,
    labels: Sequence[int],
    *,
    group_ids: Sequence[Hashable] | None = None,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
    iterations: int = 64,
) -> ScalarCalibration:
    """Fit one game-balanced inverse temperature by convex bisection.

    Labels use network class indices: win=0, draw=1, loss=2. With group IDs,
    every game contributes equal total weight regardless of its number of plies.
    """

    rows = _rows(logits)
    if len(labels) != len(rows) or any(label not in (0, 1, 2) for label in labels):
        raise ValueError("calibration labels must match logits and be in {0, 1, 2}")
    if (
        not math.isfinite(minimum_scale)
        or not math.isfinite(maximum_scale)
        or not 0 < minimum_scale < maximum_scale
        or iterations <= 0
    ):
        raise ValueError("scalar calibration bounds are invalid")
    weights, groups = _weights(len(rows), group_ids)
    before, _ = _objective(rows, labels, weights, 1.0)
    low_loss, low_derivative = _objective(rows, labels, weights, minimum_scale)
    high_loss, high_derivative = _objective(rows, labels, weights, maximum_scale)
    if low_derivative >= 0:
        scale, after = minimum_scale, low_loss
        used_iterations = 0
    elif high_derivative <= 0:
        scale, after = maximum_scale, high_loss
        used_iterations = 0
    else:
        low, high = minimum_scale, maximum_scale
        for _ in range(iterations):
            middle = (low + high) / 2.0
            _, derivative = _objective(rows, labels, weights, middle)
            if derivative < 0:
                low = middle
            else:
                high = middle
        scale = (low + high) / 2.0
        after, _ = _objective(rows, labels, weights, scale)
        used_iterations = iterations
    return ScalarCalibration(
        logit_scale=scale,
        temperature=1.0 / scale,
        fit_cross_entropy_before=before,
        fit_cross_entropy_after=after,
        iterations=used_iterations,
        rows=len(rows),
        groups=groups,
    )


def scaled_logits(logits: mx.array, calibration: ScalarCalibration) -> mx.array:
    return logits * calibration.logit_scale
