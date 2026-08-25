"""Pre-registered paired strength and behavior gate for arena ablations."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PairedGateConfig:
    confidence_level: float = 0.95
    avoidable_noninferiority_margin: float = 0.05
    win_rate_noninferiority_margin: float = 0.02
    bootstrap_samples: int = 50_000
    seed: int = 2026082506

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("paired gate confidence level must be in (0, 1)")
        if not 0.0 <= self.avoidable_noninferiority_margin <= 1.0:
            raise ValueError("avoidable repetition margin must be in [0, 1]")
        if not 0.0 <= self.win_rate_noninferiority_margin <= 1.0:
            raise ValueError("win-rate margin must be in [0, 1]")
        if self.bootstrap_samples <= 0 or self.seed < 0:
            raise ValueError("paired gate bootstrap configuration is invalid")


@dataclass(frozen=True, slots=True)
class PairedObservation:
    control_score: float
    candidate_score: float
    control_avoidable: bool
    candidate_avoidable: bool

    def __post_init__(self) -> None:
        if self.control_score not in (0.0, 0.5, 1.0) or self.candidate_score not in (
            0.0,
            0.5,
            1.0,
        ):
            raise ValueError("paired arena scores must be loss, draw, or win")


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class PairedGateResult:
    games: int
    strength: Interval
    avoidable_rate: Interval
    win_rate: Interval
    decisive_score_control: float
    decisive_score_candidate: float
    decisive_score_delta: Interval
    strength_passed: bool
    avoidable_passed: bool
    win_rate_passed: bool
    decisive_passed: bool
    passed: bool


def _quantile(sorted_values: list[float], probability: float) -> float:
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def _decisive_score(scores: list[float]) -> float:
    decisive = [score for score in scores if score != 0.5]
    return sum(decisive) / len(decisive) if decisive else 0.5


def evaluate_paired_gate(
    observations: tuple[PairedObservation, ...],
    *,
    config: PairedGateConfig | None = None,
) -> PairedGateResult:
    settings = config or PairedGateConfig()
    if not observations:
        raise ValueError("paired gate requires observations")
    score_deltas = [item.candidate_score - item.control_score for item in observations]
    avoidable_deltas = [
        float(item.candidate_avoidable) - float(item.control_avoidable) for item in observations
    ]
    win_deltas = [
        float(item.candidate_score == 1.0) - float(item.control_score == 1.0)
        for item in observations
    ]
    control_scores = [item.control_score for item in observations]
    candidate_scores = [item.candidate_score for item in observations]
    score_estimate = sum(score_deltas) / len(observations)
    avoidable_estimate = sum(avoidable_deltas) / len(observations)
    win_estimate = sum(win_deltas) / len(observations)
    decisive_control = _decisive_score(control_scores)
    decisive_candidate = _decisive_score(candidate_scores)
    decisive_estimate = decisive_candidate - decisive_control

    rng = random.Random(settings.seed)
    score_bootstrap = []
    avoidable_bootstrap = []
    win_bootstrap = []
    decisive_bootstrap = []
    count = len(observations)
    for _ in range(settings.bootstrap_samples):
        indices = [rng.randrange(count) for _ in range(count)]
        score_bootstrap.append(sum(score_deltas[index] for index in indices) / count)
        avoidable_bootstrap.append(sum(avoidable_deltas[index] for index in indices) / count)
        win_bootstrap.append(sum(win_deltas[index] for index in indices) / count)
        decisive_bootstrap.append(
            _decisive_score([candidate_scores[index] for index in indices])
            - _decisive_score([control_scores[index] for index in indices])
        )
    score_bootstrap.sort()
    avoidable_bootstrap.sort()
    win_bootstrap.sort()
    decisive_bootstrap.sort()
    alpha = 1.0 - settings.confidence_level
    strength = Interval(
        score_estimate,
        _quantile(score_bootstrap, alpha / 2.0),
        _quantile(score_bootstrap, 1.0 - alpha / 2.0),
    )
    avoidable = Interval(
        avoidable_estimate,
        _quantile(avoidable_bootstrap, alpha),
        _quantile(avoidable_bootstrap, settings.confidence_level),
    )
    win_rate = Interval(
        win_estimate,
        _quantile(win_bootstrap, alpha),
        _quantile(win_bootstrap, settings.confidence_level),
    )
    decisive = Interval(
        decisive_estimate,
        _quantile(decisive_bootstrap, alpha),
        _quantile(decisive_bootstrap, settings.confidence_level),
    )
    strength_passed = strength.low > 0.0
    avoidable_passed = (
        avoidable.estimate <= 0.0 and avoidable.high <= settings.avoidable_noninferiority_margin
    )
    win_rate_passed = (
        win_rate.estimate > 0.0 and win_rate.low >= -settings.win_rate_noninferiority_margin
    )
    decisive_passed = decisive.estimate >= 0.0
    return PairedGateResult(
        games=count,
        strength=strength,
        avoidable_rate=avoidable,
        win_rate=win_rate,
        decisive_score_control=decisive_control,
        decisive_score_candidate=decisive_candidate,
        decisive_score_delta=decisive,
        strength_passed=strength_passed,
        avoidable_passed=avoidable_passed,
        win_rate_passed=win_rate_passed,
        decisive_passed=decisive_passed,
        passed=(strength_passed and avoidable_passed and win_rate_passed and decisive_passed),
    )


def observations_from_results(
    control_paths: tuple[Path, ...],
    candidate_paths: tuple[Path, ...],
) -> tuple[PairedObservation, ...]:
    if not control_paths or len(control_paths) != len(candidate_paths):
        raise ValueError("paired gate requires matched control and candidate result batches")
    observations = []
    for batch, (control_path, candidate_path) in enumerate(
        zip(control_paths, candidate_paths, strict=True)
    ):
        control = json.loads(control_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        control_games = {
            (game["pair_index"], game["candidate_side"]): game for game in control["games"]
        }
        candidate_games = {
            (game["pair_index"], game["candidate_side"]): game for game in candidate["games"]
        }
        if set(control_games) != set(candidate_games):
            raise ValueError(f"paired arena batch {batch} does not contain matching games")
        for key in sorted(control_games):
            control_game = control_games[key]
            candidate_game = candidate_games[key]
            if control_game["opening_moves"] != candidate_game["opening_moves"]:
                raise ValueError(f"paired arena batch {batch} opening mismatch at {key}")
            observations.append(
                PairedObservation(
                    control_score=float(control_game["candidate_score"]),
                    candidate_score=float(candidate_game["candidate_score"]),
                    control_avoidable=bool(control_game["avoidable_threefold"]),
                    candidate_avoidable=bool(candidate_game["avoidable_threefold"]),
                )
            )
    return tuple(observations)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-result", action="append", required=True, type=Path)
    parser.add_argument("--candidate-result", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026082506)
    parser.add_argument("--avoidable-margin", type=float, default=0.05)
    parser.add_argument("--win-rate-margin", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = PairedGateConfig(
        avoidable_noninferiority_margin=arguments.avoidable_margin,
        win_rate_noninferiority_margin=arguments.win_rate_margin,
        bootstrap_samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    observations = observations_from_results(
        tuple(arguments.control_result), tuple(arguments.candidate_result)
    )
    result = evaluate_paired_gate(observations, config=settings)
    _atomic_json(
        arguments.output,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "config": asdict(settings),
            "control_results": [str(path) for path in arguments.control_result],
            "candidate_results": [str(path) for path in arguments.candidate_result],
            "result": asdict(result),
        },
    )
    print(arguments.output)
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
