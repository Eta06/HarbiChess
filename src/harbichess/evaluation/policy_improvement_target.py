"""Build and qualify KL-constrained policy targets from stable search-Q values."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.consensus_target import _record_index, _record_policy, _top_actions
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.shard import read_shard

Policy = dict[str, float]


@dataclass(frozen=True, slots=True)
class PolicyImprovementTargetConfig:
    label_result: Path
    dataset_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    maximum_kl: float = 0.10
    harmful_delta: float = -0.025
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_effective_action_ratio: float = 0.50
    minimum_labelable_ratio: float = 0.95
    bootstrap_samples: int = 2_000
    seed: int = 2026082837
    q_mode: Literal["average", "conservative"] = "average"

    def __post_init__(self) -> None:
        if (
            not 0 < self.maximum_kl <= 1
            or self.harmful_delta > 0
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_effective_action_ratio <= 1
            or not 0 <= self.minimum_labelable_ratio <= 1
            or min(self.bootstrap_samples, self.seed) <= 0
            or self.q_mode not in {"average", "conservative"}
        ):
            raise ValueError("policy improvement target configuration is invalid")


def _normalize(values: Mapping[str, float]) -> Policy:
    total = sum(values.values())
    if total <= 0 or any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("policy values must be finite and non-negative")
    return {action: value / total for action, value in values.items() if value > 0}


def _kl(target: Mapping[str, float], raw: Mapping[str, float]) -> float:
    return sum(
        probability * math.log(probability / raw[action])
        for action, probability in target.items()
    )


def _entropy(policy: Mapping[str, float]) -> float:
    return -sum(probability * math.log(probability) for probability in policy.values())


def mirror_descent_target(
    raw_policy: Mapping[str, float],
    stable_q: Mapping[str, float],
    *,
    maximum_kl: float,
) -> tuple[Policy, float, float]:
    raw = _normalize(raw_policy)
    if not stable_q or stable_q.keys() - raw.keys():
        raise ValueError("stable Q support must be non-empty and contained in raw policy")
    stable_mass = sum(raw[action] for action in stable_q)
    anchor = sum(raw[action] * stable_q[action] for action in stable_q) / stable_mass

    def target(temperature: float) -> Policy:
        scores = {
            action: math.log(probability)
            + ((stable_q[action] - anchor) / temperature if action in stable_q else 0.0)
            for action, probability in raw.items()
        }
        maximum = max(scores.values())
        return _normalize({action: math.exp(score - maximum) for action, score in scores.items()})

    low = 1e-4
    low_target = target(low)
    if _kl(low_target, raw) <= maximum_kl:
        return low_target, _kl(low_target, raw), low
    high = 1.0
    while _kl(target(high), raw) > maximum_kl:
        high *= 2
        if high > 1e6:
            raise RuntimeError("failed to satisfy policy KL trust region")
    for _ in range(80):
        middle = math.sqrt(low * high)
        if _kl(target(middle), raw) > maximum_kl:
            low = middle
        else:
            high = middle
    result = target(high)
    return result, _kl(result, raw), high


def _expected(policy: Mapping[str, float], values: Mapping[str, float]) -> float:
    missing = policy.keys() - values.keys()
    if missing:
        raise ValueError(f"verifier values are missing actions: {sorted(missing)}")
    return sum(probability * values[action] for action, probability in policy.items())


def _row(
    label: Mapping[str, object],
    verifier: Mapping[str, object],
    raw: Mapping[str, float],
    *,
    config: PolicyImprovementTargetConfig,
) -> dict[str, object]:
    average_q = {
        str(action): float(q)
        for action, q, _drift, confidence in label["labels"]
        if float(confidence) > 0
    }
    if config.q_mode == "average":
        stable_q = average_q
    else:
        low_q = dict(verifier["budgets"]["512"]["q"])
        high_q = dict(verifier["budgets"]["800"]["q"])
        stable_q = {
            action: min(float(low_q[action]), float(high_q[action]))
            for action in average_q
        }
    confidence = {
        str(action): float(weight)
        for action, _q, _drift, weight in label["labels"]
        if float(weight) > 0
    }
    values = {str(action): float(value) for action, value in verifier["verified_values"]}
    target, kl, temperature = mirror_descent_target(
        raw, stable_q, maximum_kl=config.maximum_kl
    )
    raw_expected = _expected(raw, values)
    target_expected = _expected(target, values)
    delta = target_expected - raw_expected
    target_top = _top_actions(target, 1)[0]
    raw_top = _top_actions(raw, 1)[0]
    best = max(values.values())
    effective_ratio = math.exp(_entropy(target)) / math.exp(_entropy(raw))
    return {
        "partition": label["partition"],
        "game_id": label["game_id"],
        "game_index": label["game_index"],
        "ply": label["ply"],
        "target": tuple(sorted(target.items())),
        "raw_policy": tuple(sorted(raw.items())),
        "stable_q": tuple(sorted(stable_q.items())),
        "confidence": tuple(sorted(confidence.items())),
        "temperature": temperature,
        "target_to_raw_kl": kl,
        "effective_action_ratio": effective_ratio,
        "verified_expected": {"raw": raw_expected, "target": target_expected},
        "verified_expected_delta": delta,
        "target_top_action": target_top,
        "target_top_verified_delta_vs_raw_top": values[target_top] - values[raw_top],
        "target_top_verified_regret": best - values[target_top],
    }


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    source_positions: int,
    config: PolicyImprovementTargetConfig,
    seed: int,
) -> dict[str, object]:
    deltas = tuple(float(row["verified_expected_delta"]) for row in rows)
    top_deltas = tuple(float(row["target_top_verified_delta_vs_raw_top"]) for row in rows)
    return {
        "source_positions": source_positions,
        "positions": len(rows),
        "labelable_ratio": len(rows) / source_positions,
        "mean_target_to_raw_kl": mean(float(row["target_to_raw_kl"]) for row in rows),
        "maximum_target_to_raw_kl": max(float(row["target_to_raw_kl"]) for row in rows),
        "mean_effective_action_ratio": mean(
            float(row["effective_action_ratio"]) for row in rows
        ),
        "mean_verified_expected_delta": mean(deltas),
        "verified_expected_delta_95_interval": _interval(
            deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "harmful_expected_row_count": sum(delta <= config.harmful_delta for delta in deltas),
        "harmful_expected_row_ratio": sum(delta <= config.harmful_delta for delta in deltas)
        / len(rows),
        "target_top_harmful_count": sum(
            delta <= config.harmful_delta for delta in top_deltas
        ),
        "target_top_harmful_ratio": sum(
            delta <= config.harmful_delta for delta in top_deltas
        )
        / len(rows),
        "mean_target_top_verified_regret": mean(
            float(row["target_top_verified_regret"]) for row in rows
        ),
    }


def _gate(
    summary: Mapping[str, object], config: PolicyImprovementTargetConfig
) -> dict[str, object]:
    reasons = []
    if float(summary["labelable_ratio"]) < config.minimum_labelable_ratio:
        reasons.append("policy target labelable ratio is below 95%")
    if float(summary["verified_expected_delta_95_interval"][0]) <= 0:
        reasons.append("verified expected-value improvement interval is not positive")
    if float(summary["maximum_target_to_raw_kl"]) > config.maximum_kl + 1e-9:
        reasons.append("policy target exceeds the KL trust region")
    if float(summary["harmful_expected_row_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("harmful expected-value row ratio exceeds 10%")
    if float(summary["target_top_harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("target-top harmful-action ratio exceeds 10%")
    if float(summary["mean_target_top_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("target-top mean verified regret exceeds 0.10")
    if float(summary["mean_effective_action_ratio"]) < config.minimum_effective_action_ratio:
        reasons.append("target effective-action ratio is below 50%")
    return {"passed": not reasons, "reasons": reasons}


def run_policy_improvement_target(config: PolicyImprovementTargetConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"policy improvement output exists: {config.output_dir}")
    labels = json.loads(config.label_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    if not labels.get("gate", {}).get("passed"):
        raise ValueError("policy improvement requires qualified uncertainty labels")
    rules = PythonChessRules()
    records = {
        "train": _record_index(read_shard(config.train_shard, rules=rules).records),
        "validation": _record_index(
            read_shard(config.validation_shard, rules=rules).records
        ),
    }
    output = {}
    summaries = {}
    for index, partition in enumerate(("train", "validation")):
        verifier = {
            (str(row["game_id"]), int(row["game_index"]), int(row["ply"])): row
            for row in dataset["rows"][partition]
        }
        rows = []
        for label in labels["rows"][partition]:
            key = (str(label["game_id"]), int(label["game_index"]), int(label["ply"]))
            record = records[partition].get(key)
            verifier_row = verifier.get(key)
            if record is None or verifier_row is None:
                raise ValueError(f"policy target row is absent from source data: {key}")
            rows.append(
                _row(
                    label,
                    verifier_row,
                    _record_policy(record, rules),
                    config=config,
                )
            )
        output[partition] = tuple(rows)
        summaries[partition] = _summary(
            rows,
            source_positions=int(labels["summaries"][partition]["source_positions"]),
            config=config,
            seed=config.seed + index * 10,
        )
    gate = _gate(summaries["validation"], config)
    result_path = config.output_dir / "targets.json"
    _atomic_json(
        result_path,
        {
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "label_result": str(config.label_result),
                "dataset_result": str(config.dataset_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "learner_transfer_authorized": gate["passed"],
                "arena_authorized": False,
                "generation_authorized": False,
                "promotion_authorized": False,
            },
            "rows": output,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-result", required=True, type=Path)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--q-mode", choices=("average", "conservative"), default="average"
    )
    arguments = parser.parse_args(argv)
    path = run_policy_improvement_target(
        PolicyImprovementTargetConfig(
            label_result=arguments.label_result,
            dataset_result=arguments.dataset_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
            q_mode=arguments.q_mode,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
