"""Plan fresh cumulative-gate sample size from a frozen diagnostic checkpoint."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.cumulative_value_gate import PredictionGame, paired_power_plan
from harbichess.evaluation.deterministic_value_probe import _prepare
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.schema import ReplayRecord
from harbichess.training.full_gumbel_transfer import _network
from harbichess.training.joint_policy_value_transfer import _sha256
from harbichess.training.stable_plastic_ablation import (
    StablePlasticAblationConfig,
    _fresh_split,
    _load_fresh_records,
    _load_mihver,
)


@dataclass(frozen=True, slots=True)
class CumulativePowerPlanConfig:
    output_dir: Path
    value_result: Path
    model_path: Path
    source_continuous_result: Path
    pilot_candidate_path: Path
    runs_root: Path = Path("artifacts/runs")
    expected_pilot_sha256: str = (
        "077464e15e71121cc07976ad1daf49fa4459ffd9406a6252a39a415ae06a621f"
    )
    old_ce_margin: float = 0.003
    fresh_ce_minimum_improvement: float = 0.002
    fresh_ce_design_improvement: float = 0.006
    alpha: float = 0.05
    power: float = 0.80
    inflation: float = 0.15
    round_to: int = 24
    minimum_games: int = 192
    maximum_games: int = 1536
    seed: int = 2026090101

    def __post_init__(self) -> None:
        if (
            min(
                self.old_ce_margin,
                self.fresh_ce_minimum_improvement,
                self.fresh_ce_design_improvement,
                self.round_to,
                self.minimum_games,
                self.maximum_games,
                self.seed,
            )
            <= 0
            or not 0.0 < self.alpha < 0.5
            or not 0.5 < self.power < 1.0
            or self.inflation < 0.0
            or self.fresh_ce_design_improvement <= self.fresh_ce_minimum_improvement
            or self.minimum_games > self.maximum_games
            or self.minimum_games % self.round_to
            or self.maximum_games % self.round_to
        ):
            raise ValueError("cumulative power plan configuration is invalid")
        if len(self.expected_pilot_sha256) != 64:
            raise ValueError("pilot candidate hash must be SHA-256")


def _probabilities(
    network, records: tuple[ReplayRecord, ...], rules
) -> tuple[tuple[float, ...], ...]:
    inputs, _ = _prepare(records, rules)
    probabilities = mx.softmax(network(inputs)[1], axis=1)
    mx.eval(probabilities)
    return tuple(tuple(float(value) for value in row) for row in probabilities.tolist())


def prediction_games(
    baseline,
    candidate,
    records: tuple[ReplayRecord, ...],
    *,
    rules: PythonChessRules,
) -> tuple[PredictionGame, ...]:
    baseline_probabilities = _probabilities(baseline, records, rules)
    candidate_probabilities = _probabilities(candidate, records, rules)
    grouped = defaultdict(lambda: {"outcomes": [], "baseline": [], "candidate": []})
    for record, baseline_row, candidate_row in zip(
        records, baseline_probabilities, candidate_probabilities, strict=True
    ):
        if record.outcome_value is None:
            continue
        group = grouped[record.game_id]
        group["outcomes"].append(int(record.outcome_value))
        group["baseline"].append(baseline_row)
        group["candidate"].append(candidate_row)
    return tuple(
        PredictionGame(
            game_id=game_id,
            outcomes=tuple(group["outcomes"]),
            baseline_probabilities=tuple(group["baseline"]),
            candidate_probabilities=tuple(group["candidate"]),
        )
        for game_id, group in sorted(grouped.items())
    )


def _game_ce_difference(game: PredictionGame, *, improvement: bool) -> float:
    labels = ({1: 0, 0: 1, -1: 2}[outcome] for outcome in game.outcomes)
    differences = []
    for label, baseline, candidate in zip(
        labels,
        game.baseline_probabilities,
        game.candidate_probabilities,
        strict=True,
    ):
        baseline_loss = -math.log(max(baseline[label], 1e-12))
        candidate_loss = -math.log(max(candidate[label], 1e-12))
        differences.append(
            baseline_loss - candidate_loss if improvement else candidate_loss - baseline_loss
        )
    return statistics.fmean(differences)


def run_cumulative_power_plan(config: CumulativePowerPlanConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"power plan output exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    if _sha256(config.pilot_candidate_path) != config.expected_pilot_sha256:
        raise ValueError("pilot candidate hash does not match frozen planning input")
    baseline = _load_mihver(
        StablePlasticAblationConfig(
            output_dir=config.output_dir,
            value_result=config.value_result,
            model_path=config.model_path,
            source_continuous_result=config.source_continuous_result,
            runs_root=config.runs_root,
        )
    )
    candidate = HarbiChessDecoupledValueNetwork.from_base(_network())
    candidate.load_weights(str(config.pilot_candidate_path))
    mx.eval(candidate.parameters())
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    _, old_validation, old_split = _split_games(games, seed=pool_config.seed)
    fresh_records, fresh_paths = _load_fresh_records(
        StablePlasticAblationConfig(
            output_dir=config.output_dir,
            value_result=config.value_result,
            model_path=config.model_path,
            source_continuous_result=config.source_continuous_result,
            runs_root=config.runs_root,
        )
    )
    _, fresh_validation, fresh_split = _fresh_split(fresh_records, seed=2026083111)
    rules = PythonChessRules()
    old_games = prediction_games(baseline, candidate, old_validation, rules=rules)
    fresh_games = prediction_games(baseline, candidate, fresh_validation, rules=rules)
    old_differences = tuple(
        _game_ce_difference(game, improvement=False) for game in old_games
    )
    fresh_differences = tuple(
        _game_ce_difference(game, improvement=True) for game in fresh_games
    )
    old_standard_deviation = statistics.stdev(old_differences)
    fresh_standard_deviation = statistics.stdev(fresh_differences)
    old_plan = paired_power_plan(
        standard_deviation=old_standard_deviation,
        null_boundary=config.old_ce_margin,
        assumed_effect=0.0,
        alpha=config.alpha,
        power=config.power,
        inflation=config.inflation,
        round_to=config.round_to,
    )
    fresh_plan = paired_power_plan(
        standard_deviation=fresh_standard_deviation,
        null_boundary=config.fresh_ce_minimum_improvement,
        assumed_effect=config.fresh_ce_design_improvement,
        alpha=config.alpha,
        power=config.power,
        inflation=config.inflation,
        round_to=config.round_to,
    )
    required_games = max(
        config.minimum_games,
        old_plan.rounded_games,
        fresh_plan.rounded_games,
    )
    feasible = required_games <= config.maximum_games
    selected_games = min(required_games, config.maximum_games)
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": {
                **asdict(config),
                **{
                    key: str(getattr(config, key))
                    for key in (
                        "output_dir",
                        "value_result",
                        "model_path",
                        "source_continuous_result",
                        "pilot_candidate_path",
                        "runs_root",
                    )
                },
            },
            "planning_only": True,
            "historical_results_reclassified": False,
            "provenance": provenance,
            "old_split": old_split,
            "fresh_paths": fresh_paths,
            "fresh_split": fresh_split,
            "old": {
                "games": len(old_games),
                "observed_mean_delta": statistics.fmean(old_differences),
                "standard_deviation": old_standard_deviation,
                "power_plan": asdict(old_plan),
            },
            "fresh": {
                "games": len(fresh_games),
                "observed_mean_improvement": statistics.fmean(fresh_differences),
                "standard_deviation": fresh_standard_deviation,
                "power_plan": asdict(fresh_plan),
            },
            "required_games": required_games,
            "selected_games": selected_games,
            "feasible": feasible,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-continuous-result", required=True, type=Path)
    parser.add_argument("--pilot-candidate", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    arguments = parser.parse_args(argv)
    result = run_cumulative_power_plan(
        CumulativePowerPlanConfig(
            output_dir=arguments.output_dir,
            value_result=arguments.value_result,
            model_path=arguments.model,
            source_continuous_result=arguments.source_continuous_result,
            pilot_candidate_path=arguments.pilot_candidate,
            runs_root=arguments.runs_root,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
