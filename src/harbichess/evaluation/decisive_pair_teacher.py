"""Qualify uncertainty-aware search-Q ordering on verifier-separated pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit


@dataclass(frozen=True, slots=True)
class DecisivePairTeacherConfig:
    dataset_result: Path
    label_result: Path
    output_dir: Path
    minimum_verifier_margin: float = 0.05
    minimum_informative_position_ratio: float = 0.50
    minimum_pair_concordance: float = 0.60
    minimum_pair_interval_lower: float = 0.50
    minimum_labelable_ratio: float = 0.95
    minimum_common_support: float = 0.95
    minimum_stable_visit_mass: float = 0.80
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    bootstrap_samples: int = 2_000
    seed: int = 2026082846

    def __post_init__(self) -> None:
        ratios = (
            self.minimum_informative_position_ratio,
            self.minimum_pair_concordance,
            self.minimum_pair_interval_lower,
            self.minimum_labelable_ratio,
            self.minimum_common_support,
            self.minimum_stable_visit_mass,
            self.maximum_harmful_ratio,
        )
        if (
            self.minimum_verifier_margin <= 0
            or any(not 0 <= value <= 1 for value in ratios)
            or self.maximum_verified_regret < 0
            or min(self.bootstrap_samples, self.seed) <= 0
        ):
            raise ValueError("decisive-pair teacher configuration is invalid")


def _decisive_pair_score(
    q_values: Mapping[str, float],
    verified: Mapping[str, float],
    *,
    minimum_margin: float,
) -> tuple[float | None, int]:
    actions = tuple(sorted(q_values.keys() & verified.keys()))
    scores = []
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1 :]:
            verified_delta = verified[left] - verified[right]
            if abs(verified_delta) < minimum_margin:
                continue
            q_delta = q_values[left] - q_values[right]
            if q_delta == 0:
                scores.append(0.5)
            else:
                scores.append(float((q_delta > 0) == (verified_delta > 0)))
    return (mean(scores), len(scores)) if scores else (None, 0)


def _gate(
    summary: Mapping[str, object], config: DecisivePairTeacherConfig
) -> dict[str, object]:
    reasons = []
    if float(summary["informative_position_ratio"]) < (
        config.minimum_informative_position_ratio
    ):
        reasons.append("informative-pair position ratio is below 50%")
    if float(summary["mean_decisive_pair_concordance"]) < config.minimum_pair_concordance:
        reasons.append("decisive-pair concordance is below 0.60")
    if float(summary["decisive_pair_concordance_95_interval"][0]) <= (
        config.minimum_pair_interval_lower
    ):
        reasons.append("decisive-pair concordance interval lower bound is not above 0.50")
    if float(summary["labelable_ratio"]) < config.minimum_labelable_ratio:
        reasons.append("labelable ratio is below 95%")
    if float(summary["mean_common_support_fraction"]) < config.minimum_common_support:
        reasons.append("common visited support is below 95%")
    if float(summary["mean_stable_visit_mass"]) < config.minimum_stable_visit_mass:
        reasons.append("stable visit mass is below 80%")
    if float(summary["conservative_verified_delta_95_interval"][0]) <= 0:
        reasons.append("conservative verified-improvement interval is not positive")
    if float(summary["conservative_harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("conservative harmful-action ratio exceeds 10%")
    if float(summary["mean_conservative_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("conservative mean verified regret exceeds 0.10")
    return {"passed": not reasons, "reasons": reasons}


def run_decisive_pair_teacher(config: DecisivePairTeacherConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"decisive-pair output exists: {config.output_dir}")
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    labels = json.loads(config.label_result.read_text(encoding="utf-8"))
    rows = []
    raw_by_key = {
        (str(row["game_id"]), int(row["game_index"]), int(row["ply"])): row
        for row in dataset["rows"]["validation"]
    }
    for label in labels["rows"]["validation"]:
        key = (str(label["game_id"]), int(label["game_index"]), int(label["ply"]))
        raw = raw_by_key.get(key)
        if raw is None:
            raise ValueError(f"decisive-pair row is absent: {key}")
        stable = {
            str(action)
            for action, _q, _drift, confidence in label["labels"]
            if float(confidence) > 0
        }
        low_q = dict(raw["budgets"]["512"]["q"])
        high_q = dict(raw["budgets"]["800"]["q"])
        q_values = {
            action: min(float(low_q[action]), float(high_q[action])) for action in stable
        }
        verified = {
            str(action): float(value) for action, value in raw["verified_values"]
        }
        score, pair_count = _decisive_pair_score(
            q_values,
            verified,
            minimum_margin=config.minimum_verifier_margin,
        )
        rows.append(
            {
                "game_id": label["game_id"],
                "game_index": label["game_index"],
                "ply": label["ply"],
                "decisive_pair_concordance": score,
                "decisive_pair_count": pair_count,
            }
        )

    informative = tuple(
        float(row["decisive_pair_concordance"])
        for row in rows
        if row["decisive_pair_concordance"] is not None
    )
    if not informative:
        raise ValueError("decisive-pair teacher has no informative positions")
    label_summary = labels["summaries"]["validation"]
    summary = {
        "source_positions": int(label_summary["source_positions"]),
        "labelable_positions": len(rows),
        "labelable_ratio": float(label_summary["labelable_ratio"]),
        "informative_positions": len(informative),
        "informative_position_ratio": len(informative) / len(rows),
        "decisive_pairs": sum(int(row["decisive_pair_count"]) for row in rows),
        "mean_decisive_pair_concordance": mean(informative),
        "decisive_pair_concordance_95_interval": _interval(
            informative, samples=config.bootstrap_samples, seed=config.seed
        ),
        "mean_common_support_fraction": float(
            label_summary["mean_common_support_fraction"]
        ),
        "mean_stable_visit_mass": float(label_summary["mean_stable_visit_mass"]),
        "conservative_verified_delta_95_interval": label_summary[
            "conservative_verified_delta_95_interval"
        ],
        "conservative_harmful_ratio": float(
            label_summary["conservative_harmful_ratio"]
        ),
        "mean_conservative_verified_regret": float(
            label_summary["mean_conservative_verified_regret"]
        ),
    }
    gate = _gate(summary, config)
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "dataset_result": str(config.dataset_result),
                "label_result": str(config.label_result),
                "output_dir": str(config.output_dir),
            },
            "summary": summary,
            "gate": {
                **gate,
                "policy_target_authorized": gate["passed"],
                "learner_validation_authorized": False,
                "search_qualification_authorized": False,
                "arena_authorized": False,
                "generation_authorized": False,
                "promotion_authorized": False,
            },
            "rows": rows,
        },
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--label-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026082846)
    arguments = parser.parse_args(argv)
    result = run_decisive_pair_teacher(
        DecisivePairTeacherConfig(
            dataset_result=arguments.dataset_result,
            label_result=arguments.label_result,
            output_dir=arguments.output_dir,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
