"""Qualify soft policy targets supported in the same direction by two searches."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from harbichess.chess.actions import action_to_legal_move
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.evaluation.consensus_target import (
    Policy,
    _entropy,
    _expected_value,
    _jsd,
    _normalize,
    _record_index,
    _record_policy,
    _search_policy,
    _top_actions,
    _tv,
)
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.value_oracle import ProcessTacticalOracle, TacticalOracleConfig


@dataclass(frozen=True, slots=True)
class AgreementTargetConfig:
    consistency_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    verifier_depth: int = 4
    verifier_workers: int = 8
    bootstrap_samples: int = 2_000
    seed: int = 2026082819
    maximum_anchor_tv: float = 0.20
    minimum_expected_improvement: float = 0.02
    minimum_action_improvement: float = 0.03
    maximum_action_harm: float = -0.025
    maximum_harmful_mass: float = 0.10
    minimum_qualified_ratio: float = 0.20
    maximum_harmful_row_ratio: float = 0.10
    maximum_mean_anchor_tv: float = 0.125
    maximum_value_regression_vs_800: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(self.verifier_depth, self.verifier_workers, self.bootstrap_samples, self.seed) <= 0
            or not 0 <= self.maximum_anchor_tv <= 1
            or not 0 <= self.maximum_harmful_mass <= 1
            or not 0 <= self.minimum_qualified_ratio <= 1
            or not 0 <= self.maximum_harmful_row_ratio <= 1
            or not 0 <= self.maximum_mean_anchor_tv <= 1
            or self.minimum_expected_improvement < 0
            or self.minimum_action_improvement < 0
            or self.maximum_action_harm > 0
            or self.maximum_value_regression_vs_800 < 0
        ):
            raise ValueError("agreement target configuration is invalid")


def common_direction_target(
    raw: Mapping[str, float], first: Mapping[str, float], second: Mapping[str, float]
) -> Policy:
    """Move raw mass only where both teachers agree on its direction."""

    actions = raw.keys() | first.keys() | second.keys()
    uplift = {
        action: min(
            max(first.get(action, 0.0) - raw.get(action, 0.0), 0.0),
            max(second.get(action, 0.0) - raw.get(action, 0.0), 0.0),
        )
        for action in actions
    }
    reduction = {
        action: min(
            max(raw.get(action, 0.0) - first.get(action, 0.0), 0.0),
            max(raw.get(action, 0.0) - second.get(action, 0.0), 0.0),
        )
        for action in actions
    }
    removed = sum(reduction.values())
    uplift_total = sum(uplift.values())
    if removed <= 0 or uplift_total <= 0:
        return _normalize(raw)
    return _normalize(
        {
            action: raw.get(action, 0.0)
            - reduction[action]
            + removed * uplift[action] / uplift_total
            for action in actions
        }
    )


def _row_metrics(
    row: Mapping[str, object],
    record: ReplayRecord,
    values: Mapping[str, float],
    *,
    config: AgreementTargetConfig,
) -> dict[str, object]:
    rules = PythonChessRules()
    raw = _record_policy(record, rules)
    policy_256 = _search_policy(row, 256)
    policy_512 = _search_policy(row, 512)
    policy_800 = _search_policy(row, 800)
    anchor = common_direction_target(raw, policy_256, policy_512)
    target = common_direction_target(raw, policy_512, policy_800)
    raw_top = _top_actions(raw, 1)[0]
    target_top = _top_actions(target, 1)[0]
    raw_top_value = values[raw_top]
    expected = {
        "raw": _expected_value(raw, values),
        "search_800": _expected_value(policy_800, values),
        "target": _expected_value(target, values),
    }
    improvement_mass = sum(
        probability
        for action, probability in target.items()
        if values[action] - raw_top_value >= config.minimum_action_improvement
    )
    harmful_mass = sum(
        probability
        for action, probability in target.items()
        if values[action] - raw_top_value <= config.maximum_action_harm
    )
    anchor_tv = _tv(anchor, target)
    top_two_overlap = len(set(_top_actions(anchor, 2)) & set(_top_actions(target, 2)))
    delta = expected["target"] - expected["raw"]
    reasons = []
    if anchor_tv > config.maximum_anchor_tv:
        reasons.append("anchor-to-target TV exceeds limit")
    if not top_two_overlap:
        reasons.append("anchor and target top-two sets do not overlap")
    if delta < config.minimum_expected_improvement:
        reasons.append("target expected-value improvement is below minimum")
    if harmful_mass > config.maximum_harmful_mass:
        reasons.append("target harmful probability mass exceeds limit")
    return {
        "partition": row["partition"],
        "game_id": row["game_id"],
        "game_index": row["game_index"],
        "ply": row["ply"],
        "qualified": not reasons,
        "qualification_reasons": reasons,
        "anchor_to_target_tv": anchor_tv,
        "anchor_to_target_jsd": _jsd(anchor, target),
        "top_two_overlap": top_two_overlap,
        "target_tv_from_raw": _tv(raw, target),
        "target_entropy": _entropy(target),
        "target_effective_actions": math.exp(_entropy(target)),
        "verified_expected_value": expected,
        "target_expected_delta_vs_raw": delta,
        "target_expected_delta_vs_800": expected["target"] - expected["search_800"],
        "target_top_action": target_top,
        "target_top_verified_delta_vs_raw_top": values[target_top] - raw_top_value,
        "improvement_probability_mass": improvement_mass,
        "harmful_probability_mass": harmful_mass,
        "policies": {
            "raw": tuple(sorted(raw.items())),
            "search_800": tuple(sorted(policy_800.items())),
            "target": tuple(sorted(target.items())),
        },
    }


def _summary(
    rows: tuple[Mapping[str, object], ...], *, config: AgreementTargetConfig, seed: int
) -> dict[str, object]:
    qualified = tuple(row for row in rows if row["qualified"])
    deltas = tuple(float(row["target_expected_delta_vs_raw"]) for row in rows)
    qualified_deltas = tuple(float(row["target_expected_delta_vs_raw"]) for row in qualified)
    deltas_vs_800 = tuple(float(row["target_expected_delta_vs_800"]) for row in rows)
    harmful = sum(delta <= config.maximum_action_harm for delta in deltas)
    return {
        "positions": len(rows),
        "qualified_count": len(qualified),
        "qualified_ratio": len(qualified) / len(rows),
        "harmful_row_count": harmful,
        "harmful_row_ratio": harmful / len(rows),
        "mean_anchor_to_target_tv": mean(float(row["anchor_to_target_tv"]) for row in rows),
        "mean_anchor_to_target_jsd": mean(
            float(row["anchor_to_target_jsd"]) for row in rows
        ),
        "mean_target_tv_from_raw": mean(float(row["target_tv_from_raw"]) for row in rows),
        "mean_expected_delta_vs_raw": mean(deltas),
        "expected_delta_vs_raw_95_interval": _interval(
            deltas, samples=config.bootstrap_samples, seed=seed
        ),
        "qualified_mean_expected_delta_vs_raw": mean(qualified_deltas) if qualified_deltas else 0.0,
        "qualified_expected_delta_95_interval": _interval(
            qualified_deltas, samples=config.bootstrap_samples, seed=seed + 1
        ),
        "mean_expected_delta_vs_800": mean(deltas_vs_800),
        "mean_improvement_probability_mass": mean(
            float(row["improvement_probability_mass"]) for row in rows
        ),
        "mean_harmful_probability_mass": mean(
            float(row["harmful_probability_mass"]) for row in rows
        ),
    }


def _gate(summary: Mapping[str, object], config: AgreementTargetConfig) -> dict[str, object]:
    reasons = []
    if float(summary["qualified_ratio"]) < config.minimum_qualified_ratio:
        reasons.append("qualified target ratio is below 20%")
    if float(summary["harmful_row_ratio"]) > config.maximum_harmful_row_ratio:
        reasons.append("harmful target row ratio exceeds 10%")
    if float(summary["expected_delta_vs_raw_95_interval"][0]) <= 0:
        reasons.append("full target expected-value interval is not positive")
    if float(summary["qualified_expected_delta_95_interval"][0]) <= 0:
        reasons.append("qualified target expected-value interval is not positive")
    if float(summary["mean_anchor_to_target_tv"]) > config.maximum_mean_anchor_tv:
        reasons.append("mean anchor-to-target TV exceeds 0.125")
    if float(summary["mean_expected_delta_vs_800"]) < -config.maximum_value_regression_vs_800:
        reasons.append("target expected value regresses 800-search by more than 0.01")
    return {"passed": not reasons, "reasons": reasons}


def run_agreement_target_audit(config: AgreementTargetConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"agreement target output already exists: {config.output_dir}")
    source = json.loads(config.consistency_result.read_text(encoding="utf-8"))
    if source.get("gate", {}).get("passed"):
        raise ValueError("agreement audit expects the failed MIHENK consistency artifact")
    rules = PythonChessRules()
    records = {
        "train": _record_index(read_shard(config.train_shard, rules=rules).records),
        "validation": _record_index(read_shard(config.validation_shard, rules=rules).records),
    }
    verifier = ProcessTacticalOracle(
        TacticalOracleConfig(depth=config.verifier_depth), workers=config.verifier_workers
    )
    started = time.perf_counter()
    output_rows: dict[str, tuple[dict[str, object], ...]] = {}
    try:
        for partition in ("train", "validation"):
            work = []
            matched = []
            for row in source["rows"][partition]:
                key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
                record = records[partition].get(key)
                if record is None:
                    raise ValueError(f"MIHENK row is absent from replay: {key}")
                matched.append((row, record))
                board = rules.board(record.state)
                for action, _ in record.raw_policy:
                    move = action_to_legal_move(board, action).uci()
                    child = rules.apply(record.state, ChessMove(move))
                    work.append((len(matched) - 1, move, child))

            def verify(item: tuple[int, str, object]):
                index, action, child = item
                return index, action, -verifier.value(child)

            with ThreadPoolExecutor(max_workers=config.verifier_workers) as pool:
                verified_items = tuple(pool.map(verify, work))
            verified: dict[int, dict[str, float]] = {}
            for index, action, value in verified_items:
                verified.setdefault(index, {})[action] = value
            output_rows[partition] = tuple(
                _row_metrics(row, record, verified[index], config=config)
                for index, (row, record) in enumerate(matched)
            )
    finally:
        verifier.close()

    summaries = {
        partition: _summary(rows, config=config, seed=config.seed + index * 10)
        for index, (partition, rows) in enumerate(output_rows.items())
    }
    gate = _gate(summaries["validation"], config)
    result_path = config.output_dir / "agreement.json"
    _atomic_json(
        result_path,
        {
            "created_at": time.time(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                "consistency_result": str(config.consistency_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
            },
            "summaries": summaries,
            "gate": {
                **gate,
                "learner_ablation_authorized": gate["passed"],
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
    parser.add_argument("--consistency-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    path = run_agreement_target_audit(
        AgreementTargetConfig(
            consistency_result=arguments.consistency_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
