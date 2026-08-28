"""Qualify and materialize uncertainty-weighted cross-budget Q labels."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.consensus_target import _record_index, _record_policy, _top_actions
from harbichess.evaluation.search_q_reliability import _spearman
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.shard import read_shard


@dataclass(frozen=True, slots=True)
class UncertaintyQLabelConfig:
    dataset_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    drift_cutoff: float = 0.03
    minimum_common_support: float = 0.95
    minimum_stable_visit_mass: float = 0.80
    minimum_stable_q_verified_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    harmful_delta: float = -0.025
    maximum_verified_regret: float = 0.10
    bootstrap_samples: int = 2_000
    seed: int = 2026082828

    def __post_init__(self) -> None:
        if (
            self.drift_cutoff <= 0
            or not 0 <= self.minimum_common_support <= 1
            or not 0 <= self.minimum_stable_visit_mass <= 1
            or not -1 <= self.minimum_stable_q_verified_spearman <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.harmful_delta > 0
            or self.maximum_verified_regret < 0
            or min(self.bootstrap_samples, self.seed) <= 0
        ):
            raise ValueError("uncertainty Q-label configuration is invalid")


def uncertainty_labels(
    q_low: Mapping[str, float],
    q_high: Mapping[str, float],
    visits_low: Mapping[str, int],
    visits_high: Mapping[str, int],
    *,
    drift_cutoff: float,
) -> tuple[tuple[str, float, float, float], ...]:
    common = q_low.keys() & q_high.keys()
    rows = []
    for action in common:
        low_count = visits_low[action]
        high_count = visits_high[action]
        drift = abs(q_low[action] - q_high[action])
        target = (low_count * q_low[action] + high_count * q_high[action]) / (
            low_count + high_count
        )
        confidence = math.sqrt(min(low_count, high_count)) * max(
            0.0, 1.0 - drift / drift_cutoff
        )
        rows.append((action, target, drift, confidence))
    total = sum(row[3] for row in rows)
    if total <= 0:
        raise ValueError("uncertainty Q labels contain no confident support")
    return tuple(
        sorted(
            (action, target, drift, weight / total)
            for action, target, drift, weight in rows
        )
    )


def _row_metrics(
    row: Mapping[str, object],
    raw_policy: Mapping[str, float],
    *,
    config: UncertaintyQLabelConfig,
) -> dict[str, object]:
    low = row["budgets"]["512"]
    high = row["budgets"]["800"]
    q_low = dict(low["q"])
    q_high = dict(high["q"])
    visits_low = dict(low["visits"])
    visits_high = dict(high["visits"])
    verified = dict(row["verified_values"])
    labels = uncertainty_labels(
        q_low,
        q_high,
        visits_low,
        visits_high,
        drift_cutoff=config.drift_cutoff,
    )
    common = q_low.keys() & q_high.keys()
    stable = {
        action: (visits_low[action] * q_low[action] + visits_high[action] * q_high[action])
        / (visits_low[action] + visits_high[action])
        for action in common
        if abs(q_low[action] - q_high[action]) <= config.drift_cutoff
    }
    total_visit_weight = sum(min(visits_low[action], visits_high[action]) for action in common)
    stable_visit_weight = sum(
        min(visits_low[action], visits_high[action]) for action in stable
    )
    conservative = {action: min(q_low[action], q_high[action]) for action in common}
    selected = min(conservative, key=lambda action: (-conservative[action], action))
    raw_top = _top_actions(raw_policy, 1)[0]
    best = max(verified.values())
    return {
        "partition": row["partition"],
        "game_id": row["game_id"],
        "game_index": row["game_index"],
        "ply": row["ply"],
        "common_support_fraction": len(common) / len(verified),
        "stable_visit_mass": stable_visit_weight / total_visit_weight,
        "stable_q_verified_spearman": _spearman(stable, verified),
        "conservative_action": selected,
        "conservative_verified_delta_vs_raw": verified[selected] - verified[raw_top],
        "conservative_verified_regret": best - verified[selected],
        "labels": labels,
    }


def _summary(
    rows: tuple[Mapping[str, object], ...], *, config: UncertaintyQLabelConfig, seed: int
) -> dict[str, object]:
    deltas = tuple(float(row["conservative_verified_delta_vs_raw"]) for row in rows)
    return {
        "positions": len(rows),
        "mean_common_support_fraction": mean(
            float(row["common_support_fraction"]) for row in rows
        ),
        "mean_stable_visit_mass": mean(float(row["stable_visit_mass"]) for row in rows),
        "mean_stable_q_verified_spearman": mean(
            float(row["stable_q_verified_spearman"]) for row in rows
        ),
        "mean_conservative_verified_delta_vs_raw": mean(deltas),
        "conservative_verified_delta_95_interval": _interval(
            deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "conservative_harmful_count": sum(delta <= config.harmful_delta for delta in deltas),
        "conservative_harmful_ratio": sum(delta <= config.harmful_delta for delta in deltas)
        / len(rows),
        "mean_conservative_verified_regret": mean(
            float(row["conservative_verified_regret"]) for row in rows
        ),
    }


def _gate(summary: Mapping[str, object], config: UncertaintyQLabelConfig) -> dict[str, object]:
    reasons = []
    if float(summary["mean_common_support_fraction"]) < config.minimum_common_support:
        reasons.append("common visited support is below 95%")
    if float(summary["mean_stable_visit_mass"]) < config.minimum_stable_visit_mass:
        reasons.append("drift-qualified visit mass is below 80%")
    if float(summary["mean_stable_q_verified_spearman"]) < (
        config.minimum_stable_q_verified_spearman
    ):
        reasons.append("stable Q/verifier Spearman is below 0.35")
    if float(summary["conservative_verified_delta_95_interval"][0]) <= 0:
        reasons.append("conservative-Q verified-improvement interval is not positive")
    if float(summary["conservative_harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("conservative-Q harmful-action ratio exceeds 10%")
    if float(summary["mean_conservative_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("conservative-Q mean verified regret exceeds 0.10")
    return {"passed": not reasons, "reasons": reasons}


def run_uncertainty_q_labels(config: UncertaintyQLabelConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"uncertainty Q-label output exists: {config.output_dir}")
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    rules = PythonChessRules()
    record_indices = {
        "train": _record_index(read_shard(config.train_shard, rules=rules).records),
        "validation": _record_index(read_shard(config.validation_shard, rules=rules).records),
    }
    started = time.perf_counter()
    output_rows = {}
    for partition in ("train", "validation"):
        rows = []
        for row in dataset["rows"][partition]:
            key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
            record = record_indices[partition].get(key)
            if record is None:
                raise ValueError(f"fresh Q row is absent from replay: {key}")
            rows.append(_row_metrics(row, _record_policy(record, rules), config=config))
        output_rows[partition] = tuple(rows)
    summaries = {
        partition: _summary(rows, config=config, seed=config.seed + index * 10)
        for index, (partition, rows) in enumerate(output_rows.items())
    }
    gate = _gate(summaries["validation"], config)
    result_path = config.output_dir / "labels.json"
    _atomic_json(
        result_path,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "dataset_result": str(config.dataset_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "spatial_transfer_authorized": gate["passed"],
                "arena_authorized": False,
                "generation_authorized": False,
                "promotion_authorized": False,
            },
            "rows": output_rows,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    path = run_uncertainty_q_labels(
        UncertaintyQLabelConfig(
            dataset_result=arguments.dataset_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
