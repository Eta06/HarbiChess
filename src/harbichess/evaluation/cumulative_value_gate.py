"""Game-clustered bootstrap gates and power planning for cumulative value transfer."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class PredictionGame:
    game_id: str
    outcomes: tuple[int, ...]
    baseline_probabilities: tuple[tuple[float, float, float], ...]
    candidate_probabilities: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        count = len(self.outcomes)
        if (
            not self.game_id
            or count == 0
            or len(self.baseline_probabilities) != count
            or len(self.candidate_probabilities) != count
            or any(outcome not in (-1, 0, 1) for outcome in self.outcomes)
        ):
            raise ValueError("paired prediction game is incomplete")
        for probabilities in (
            *self.baseline_probabilities,
            *self.candidate_probabilities,
        ):
            if (
                len(probabilities) != 3
                or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
            ):
                raise ValueError("WDL probabilities must be finite and normalized")


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class CumulativeGateConfig:
    confidence_level: float = 0.95
    bootstrap_samples: int = 20_000
    seed: int = 2026090101
    old_ce_margin: float = 0.003
    old_macro_ce_margin: float = 0.005
    old_brier_margin: float = 0.003
    old_pearson_margin: float = 0.010
    old_ece_margin: float = 0.010
    old_ece_absolute_maximum: float = 0.120
    fresh_ce_minimum_improvement: float = 0.002
    fresh_macro_ce_minimum_improvement: float = 0.0
    fresh_brier_minimum_improvement: float = 0.0
    fresh_pearson_minimum_improvement: float = 0.0
    fresh_ece_margin: float = 0.020
    fresh_ece_absolute_maximum: float = 0.150

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0 or self.bootstrap_samples <= 0:
            raise ValueError("cumulative gate confidence configuration is invalid")
        margins = (
            self.old_ce_margin,
            self.old_macro_ce_margin,
            self.old_brier_margin,
            self.old_pearson_margin,
            self.old_ece_margin,
            self.old_ece_absolute_maximum,
            self.fresh_ce_minimum_improvement,
            self.fresh_macro_ce_minimum_improvement,
            self.fresh_brier_minimum_improvement,
            self.fresh_pearson_minimum_improvement,
            self.fresh_ece_margin,
            self.fresh_ece_absolute_maximum,
        )
        if any(value < 0.0 for value in margins):
            raise ValueError("cumulative gate margins must be non-negative")


@dataclass(frozen=True, slots=True)
class PowerPlan:
    standard_deviation: float
    null_boundary: float
    assumed_effect: float
    raw_games: int
    inflated_games: int
    rounded_games: int
    alpha: float
    power: float


@dataclass(slots=True)
class _Summary:
    count: int = 0
    loss_sum: float = 0.0
    brier_sum: float = 0.0
    class_loss_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    class_count: list[int] = field(default_factory=lambda: [0, 0, 0])
    expected_sum: float = 0.0
    expected_sq_sum: float = 0.0
    outcome_sum: float = 0.0
    outcome_sq_sum: float = 0.0
    expected_outcome_sum: float = 0.0
    bin_count: list[int] = field(default_factory=lambda: [0] * 10)
    bin_confidence_sum: list[float] = field(default_factory=lambda: [0.0] * 10)
    bin_correct_sum: list[float] = field(default_factory=lambda: [0.0] * 10)

    def add(self, other: _Summary) -> None:
        self.count += other.count
        self.loss_sum += other.loss_sum
        self.brier_sum += other.brier_sum
        self.expected_sum += other.expected_sum
        self.expected_sq_sum += other.expected_sq_sum
        self.outcome_sum += other.outcome_sum
        self.outcome_sq_sum += other.outcome_sq_sum
        self.expected_outcome_sum += other.expected_outcome_sum
        for index in range(3):
            self.class_loss_sum[index] += other.class_loss_sum[index]
            self.class_count[index] += other.class_count[index]
        for index in range(10):
            self.bin_count[index] += other.bin_count[index]
            self.bin_confidence_sum[index] += other.bin_confidence_sum[index]
            self.bin_correct_sum[index] += other.bin_correct_sum[index]


def _label(outcome: int) -> int:
    return {1: 0, 0: 1, -1: 2}[outcome]


def _summarize_game(game: PredictionGame, *, candidate: bool) -> _Summary:
    summary = _Summary()
    probabilities = game.candidate_probabilities if candidate else game.baseline_probabilities
    for outcome, probability in zip(game.outcomes, probabilities, strict=True):
        label = _label(outcome)
        loss = -math.log(max(probability[label], 1e-12))
        target = tuple(float(index == label) for index in range(3))
        expected = probability[0] - probability[2]
        predicted = max(range(3), key=probability.__getitem__)
        confidence = probability[predicted]
        bin_index = min(9, int(confidence * 10))
        summary.count += 1
        summary.loss_sum += loss
        summary.brier_sum += sum(
            (value - truth) ** 2
            for value, truth in zip(probability, target, strict=True)
        )
        summary.class_loss_sum[label] += loss
        summary.class_count[label] += 1
        summary.expected_sum += expected
        summary.expected_sq_sum += expected * expected
        summary.outcome_sum += outcome
        summary.outcome_sq_sum += outcome * outcome
        summary.expected_outcome_sum += expected * outcome
        summary.bin_count[bin_index] += 1
        summary.bin_confidence_sum[bin_index] += confidence
        summary.bin_correct_sum[bin_index] += float(predicted == label)
    return summary


def _quality_summaries(summaries: tuple[_Summary, ...]) -> dict[str, float]:
    total = _Summary()
    for summary in summaries:
        total.add(summary)
    mean_expected = total.expected_sum / total.count
    mean_outcome = total.outcome_sum / total.count
    covariance = total.expected_outcome_sum - total.count * mean_expected * mean_outcome
    expected_variance = total.expected_sq_sum - total.count * mean_expected**2
    outcome_variance = total.outcome_sq_sum - total.count * mean_outcome**2
    denominator = math.sqrt(max(0.0, expected_variance * outcome_variance))
    pearson = covariance / denominator if denominator else 0.0
    ece = sum(
        count
        / total.count
        * abs(
            total.bin_correct_sum[index] / count
            - total.bin_confidence_sum[index] / count
        )
        for index, count in enumerate(total.bin_count)
        if count
    )
    return {
        "cross_entropy": total.loss_sum / total.count,
        "macro_cross_entropy": statistics.fmean(
            total.class_loss_sum[index] / count
            for index, count in enumerate(total.class_count)
            if count
        ),
        "brier": total.brier_sum / total.count,
        "pearson": pearson,
        "ece_10": ece,
    }


def _quality(games: tuple[PredictionGame, ...], *, candidate: bool) -> dict[str, float]:
    return _quality_summaries(
        tuple(_summarize_game(game, candidate=candidate) for game in games)
    )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _interval(estimate: float, samples: list[float], confidence: float) -> Interval:
    alpha = 1.0 - confidence
    return Interval(estimate, _quantile(samples, alpha), _quantile(samples, confidence))


def _differences(
    baseline: dict[str, float], candidate: dict[str, float], *, improvement: bool
) -> dict[str, float]:
    if improvement:
        return {
            "cross_entropy": baseline["cross_entropy"] - candidate["cross_entropy"],
            "macro_cross_entropy": baseline["macro_cross_entropy"]
            - candidate["macro_cross_entropy"],
            "brier": baseline["brier"] - candidate["brier"],
            "pearson": candidate["pearson"] - baseline["pearson"],
            "ece_10": baseline["ece_10"] - candidate["ece_10"],
        }
    return {
        "cross_entropy": candidate["cross_entropy"] - baseline["cross_entropy"],
        "macro_cross_entropy": candidate["macro_cross_entropy"]
        - baseline["macro_cross_entropy"],
        "brier": candidate["brier"] - baseline["brier"],
        "pearson": candidate["pearson"] - baseline["pearson"],
        "ece_10": candidate["ece_10"] - baseline["ece_10"],
    }


def paired_bootstrap(
    games: tuple[PredictionGame, ...],
    *,
    improvement: bool,
    config: CumulativeGateConfig,
    seed_offset: int = 0,
) -> dict[str, object]:
    if len(games) < 2:
        raise ValueError("paired bootstrap requires at least two games")
    baseline_summaries = tuple(_summarize_game(game, candidate=False) for game in games)
    candidate_summaries = tuple(_summarize_game(game, candidate=True) for game in games)
    baseline = _quality_summaries(baseline_summaries)
    candidate = _quality_summaries(candidate_summaries)
    estimates = _differences(baseline, candidate, improvement=improvement)
    samples = {metric: [] for metric in estimates}
    rng = random.Random(config.seed + seed_offset)
    for _ in range(config.bootstrap_samples):
        indices = tuple(rng.randrange(len(games)) for _ in games)
        differences = _differences(
            _quality_summaries(tuple(baseline_summaries[index] for index in indices)),
            _quality_summaries(tuple(candidate_summaries[index] for index in indices)),
            improvement=improvement,
        )
        for metric, value in differences.items():
            samples[metric].append(value)
    return {
        "games": len(games),
        "baseline": baseline,
        "candidate": candidate,
        "intervals": {
            metric: asdict(_interval(estimate, samples[metric], config.confidence_level))
            for metric, estimate in estimates.items()
        },
    }


def evaluate_cumulative_gate(
    old_games: tuple[PredictionGame, ...],
    fresh_games: tuple[PredictionGame, ...],
    *,
    config: CumulativeGateConfig | None = None,
) -> dict[str, object]:
    settings = config or CumulativeGateConfig()
    old = paired_bootstrap(old_games, improvement=False, config=settings)
    fresh = paired_bootstrap(fresh_games, improvement=True, config=settings, seed_offset=1)
    old_intervals = old["intervals"]
    fresh_intervals = fresh["intervals"]
    checks = {
        "old_ce_noninferior": old_intervals["cross_entropy"]["high"]
        <= settings.old_ce_margin,
        "old_macro_ce_noninferior": old_intervals["macro_cross_entropy"]["high"]
        <= settings.old_macro_ce_margin,
        "old_brier_noninferior": old_intervals["brier"]["high"]
        <= settings.old_brier_margin,
        "old_pearson_noninferior": old_intervals["pearson"]["low"]
        >= -settings.old_pearson_margin,
        "old_ece_noninferior": old_intervals["ece_10"]["high"]
        <= settings.old_ece_margin,
        "old_ece_absolute": old["candidate"]["ece_10"]
        <= settings.old_ece_absolute_maximum,
        "fresh_ce_superior": fresh_intervals["cross_entropy"]["low"]
        >= settings.fresh_ce_minimum_improvement,
        "fresh_macro_ce_superior": fresh_intervals["macro_cross_entropy"]["low"] + 1e-12
        >= settings.fresh_macro_ce_minimum_improvement,
        "fresh_brier_superior": fresh_intervals["brier"]["low"] + 1e-12
        >= settings.fresh_brier_minimum_improvement,
        "fresh_pearson_superior": fresh_intervals["pearson"]["low"] + 1e-12
        >= settings.fresh_pearson_minimum_improvement,
        "fresh_ece_noninferior": fresh_intervals["ece_10"]["low"] + 1e-12
        >= -settings.fresh_ece_margin,
        "fresh_ece_absolute": fresh["candidate"]["ece_10"]
        <= settings.fresh_ece_absolute_maximum,
    }
    return {
        "config": asdict(settings),
        "old": old,
        "fresh": fresh,
        "checks": checks,
        "passed": all(checks.values()),
    }


def paired_power_plan(
    *,
    standard_deviation: float,
    null_boundary: float,
    assumed_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    inflation: float = 0.15,
    round_to: int = 24,
) -> PowerPlan:
    if (
        standard_deviation <= 0.0
        or not 0.0 < alpha < 0.5
        or not 0.5 < power < 1.0
        or inflation < 0.0
        or round_to <= 0
    ):
        raise ValueError("power planning inputs are invalid")
    gap = abs(assumed_effect - null_boundary)
    if gap <= 0.0:
        raise ValueError("assumed effect must differ from the null boundary")
    normal = statistics.NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_power = normal.inv_cdf(power)
    raw = math.ceil(((z_alpha + z_power) * standard_deviation / gap) ** 2)
    inflated = math.ceil(raw * (1.0 + inflation))
    rounded = max(round_to, math.ceil(inflated / round_to) * round_to)
    return PowerPlan(
        standard_deviation=standard_deviation,
        null_boundary=null_boundary,
        assumed_effect=assumed_effect,
        raw_games=raw,
        inflated_games=inflated,
        rounded_games=rounded,
        alpha=alpha,
        power=power,
    )
