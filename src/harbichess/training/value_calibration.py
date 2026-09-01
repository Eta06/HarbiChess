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


@dataclass(frozen=True, slots=True)
class GuardedScalarCalibration:
    """Fresh-optimal calibration clipped by an old-capability guard."""

    selected: ScalarCalibration
    unconstrained: ScalarCalibration
    guard_pearson_before: float
    guard_pearson_unconstrained: float
    guard_pearson_selected: float
    guard_pearson_margin: float
    constraint_active: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected.to_dict(),
            "unconstrained": self.unconstrained.to_dict(),
            "guard_pearson_before": self.guard_pearson_before,
            "guard_pearson_unconstrained": self.guard_pearson_unconstrained,
            "guard_pearson_selected": self.guard_pearson_selected,
            "guard_pearson_margin": self.guard_pearson_margin,
            "constraint_active": self.constraint_active,
        }


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


def _expected_score_pearson(
    rows: Sequence[Sequence[float]], outcomes: Sequence[int], scale: float
) -> float:
    expected = []
    for row in rows:
        scaled = tuple(scale * value for value in row)
        maximum = max(scaled)
        exponentials = tuple(math.exp(value - maximum) for value in scaled)
        total = sum(exponentials)
        expected.append((exponentials[0] - exponentials[2]) / total)
    mean_expected = sum(expected) / len(expected)
    mean_outcome = sum(outcomes) / len(outcomes)
    centered_expected = tuple(value - mean_expected for value in expected)
    centered_outcome = tuple(value - mean_outcome for value in outcomes)
    denominator = math.sqrt(
        sum(value * value for value in centered_expected)
        * sum(value * value for value in centered_outcome)
    )
    if not denominator:
        return 0.0
    return sum(
        left * right
        for left, right in zip(centered_expected, centered_outcome, strict=True)
    ) / denominator


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


def fit_guarded_scalar_calibration(
    fit_logits: mx.array,
    fit_labels: Sequence[int],
    guard_logits: mx.array,
    guard_outcomes: Sequence[int],
    *,
    fit_group_ids: Sequence[Hashable] | None = None,
    guard_pearson_margin: float = 0.005,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
    iterations: int = 64,
) -> GuardedScalarCalibration:
    """Clip fresh CE-optimal scale at an old-distribution Pearson margin."""

    if not math.isfinite(guard_pearson_margin) or guard_pearson_margin < 0:
        raise ValueError("guard Pearson margin must be finite and non-negative")
    unconstrained = fit_scalar_calibration(
        fit_logits,
        fit_labels,
        group_ids=fit_group_ids,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        iterations=iterations,
    )
    fit_rows = _rows(fit_logits)
    fit_weights, fit_groups = _weights(len(fit_rows), fit_group_ids)
    guard_rows = _rows(guard_logits)
    if len(guard_outcomes) != len(guard_rows) or any(
        outcome not in (-1, 0, 1) for outcome in guard_outcomes
    ):
        raise ValueError("guard outcomes must match logits and be in {-1, 0, 1}")
    before = _expected_score_pearson(guard_rows, guard_outcomes, 1.0)
    unconstrained_pearson = _expected_score_pearson(
        guard_rows, guard_outcomes, unconstrained.logit_scale
    )
    minimum_pearson = before - guard_pearson_margin
    if unconstrained_pearson >= minimum_pearson:
        return GuardedScalarCalibration(
            selected=unconstrained,
            unconstrained=unconstrained,
            guard_pearson_before=before,
            guard_pearson_unconstrained=unconstrained_pearson,
            guard_pearson_selected=unconstrained_pearson,
            guard_pearson_margin=guard_pearson_margin,
            constraint_active=False,
        )

    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        scale = 1.0 + middle * (unconstrained.logit_scale - 1.0)
        pearson = _expected_score_pearson(guard_rows, guard_outcomes, scale)
        if pearson >= minimum_pearson:
            low = middle
        else:
            high = middle
    selected_scale = 1.0 + low * (unconstrained.logit_scale - 1.0)
    selected_loss, _ = _objective(
        fit_rows, fit_labels, fit_weights, selected_scale
    )
    before_loss, _ = _objective(fit_rows, fit_labels, fit_weights, 1.0)
    selected = ScalarCalibration(
        logit_scale=selected_scale,
        temperature=1.0 / selected_scale,
        fit_cross_entropy_before=before_loss,
        fit_cross_entropy_after=selected_loss,
        iterations=iterations,
        rows=len(fit_rows),
        groups=fit_groups,
    )
    return GuardedScalarCalibration(
        selected=selected,
        unconstrained=unconstrained,
        guard_pearson_before=before,
        guard_pearson_unconstrained=unconstrained_pearson,
        guard_pearson_selected=_expected_score_pearson(
            guard_rows, guard_outcomes, selected_scale
        ),
        guard_pearson_margin=guard_pearson_margin,
        constraint_active=True,
    )


def scaled_logits(logits: mx.array, calibration: ScalarCalibration) -> mx.array:
    return logits * calibration.logit_scale
