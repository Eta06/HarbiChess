"""Qualify uncertainty-preserving high-budget search targets."""

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
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.value_oracle import ProcessTacticalOracle, TacticalOracleConfig

Policy = dict[str, float]


@dataclass(frozen=True, slots=True)
class ConsensusTargetConfig:
    consistency_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    verifier_depth: int = 4
    verifier_workers: int = 8
    bootstrap_samples: int = 2_000
    seed: int = 2026082818
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
            raise ValueError("consensus target configuration is invalid")


def _normalize(policy: Mapping[str, float]) -> Policy:
    total = sum(policy.values())
    if total <= 0 or any(not math.isfinite(value) or value < 0 for value in policy.values()):
        raise ValueError("policy must contain finite non-negative mass")
    return {action: value / total for action, value in policy.items() if value > 0}


def _mixture(first: Mapping[str, float], second: Mapping[str, float]) -> Policy:
    actions = first.keys() | second.keys()
    return _normalize(
        {action: 0.5 * (first.get(action, 0.0) + second.get(action, 0.0)) for action in actions}
    )


def _tv(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    actions = first.keys() | second.keys()
    return 0.5 * sum(abs(first.get(action, 0.0) - second.get(action, 0.0)) for action in actions)


def _kl(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return sum(
        probability * math.log(probability / max(second.get(action, 0.0), 1e-12))
        for action, probability in first.items()
        if probability > 0
    )


def _jsd(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    mixture = _mixture(first, second)
    return 0.5 * _kl(first, mixture) + 0.5 * _kl(second, mixture)


def _entropy(policy: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in policy.values() if value > 0)


def _top_actions(policy: Mapping[str, float], count: int) -> tuple[str, ...]:
    return tuple(
        action for action, _ in sorted(policy.items(), key=lambda item: (-item[1], item[0]))[:count]
    )


def _expected_value(policy: Mapping[str, float], values: Mapping[str, float]) -> float:
    missing = policy.keys() - values.keys()
    if missing:
        raise ValueError(f"verified values are missing policy actions: {sorted(missing)}")
    return sum(probability * values[action] for action, probability in policy.items())


def _record_policy(record: ReplayRecord, rules: PythonChessRules) -> Policy:
    board = rules.board(record.state)
    return _normalize(
        {
            action_to_legal_move(board, action).uci(): probability
            for action, probability in record.raw_policy
        }
    )


def _search_policy(row: Mapping[str, object], budget: int) -> Policy:
    budgets = row["budgets"]
    assert isinstance(budgets, dict)
    payload = budgets[str(budget)]
    assert isinstance(payload, dict)
    return _normalize(dict(payload["policy"]))


def _row_metrics(
    row: Mapping[str, object],
    record: ReplayRecord,
    values: Mapping[str, float],
    *,
    config: ConsensusTargetConfig,
) -> dict[str, object]:
    rules = PythonChessRules()
    raw = _record_policy(record, rules)
    policy_256 = _search_policy(row, 256)
    policy_512 = _search_policy(row, 512)
    policy_800 = _search_policy(row, 800)
    anchor = _mixture(policy_256, policy_512)
    consensus = _mixture(policy_512, policy_800)
    raw_top = _top_actions(raw, 1)[0]
    consensus_top = _top_actions(consensus, 1)[0]
    raw_top_value = values[raw_top]
    expected = {
        "raw": _expected_value(raw, values),
        "search_512": _expected_value(policy_512, values),
        "search_800": _expected_value(policy_800, values),
        "consensus": _expected_value(consensus, values),
    }
    improvement_mass = sum(
        probability
        for action, probability in consensus.items()
        if values[action] - raw_top_value >= config.minimum_action_improvement
    )
    harmful_mass = sum(
        probability
        for action, probability in consensus.items()
        if values[action] - raw_top_value <= config.maximum_action_harm
    )
    anchor_tv = _tv(anchor, consensus)
    top_two_overlap = len(set(_top_actions(anchor, 2)) & set(_top_actions(consensus, 2)))
    delta = expected["consensus"] - expected["raw"]
    reasons = []
    if anchor_tv > config.maximum_anchor_tv:
        reasons.append("anchor-to-consensus TV exceeds limit")
    if not top_two_overlap:
        reasons.append("anchor and consensus top-two sets do not overlap")
    if delta < config.minimum_expected_improvement:
        reasons.append("consensus expected-value improvement is below minimum")
    if harmful_mass > config.maximum_harmful_mass:
        reasons.append("consensus harmful probability mass exceeds limit")
    return {
        "partition": row["partition"],
        "game_id": row["game_id"],
        "game_index": row["game_index"],
        "ply": row["ply"],
        "qualified": not reasons,
        "qualification_reasons": reasons,
        "anchor_to_consensus_tv": anchor_tv,
        "anchor_to_consensus_jsd": _jsd(anchor, consensus),
        "top_two_overlap": top_two_overlap,
        "entropy": {
            name: _entropy(policy)
            for name, policy in {
                "raw": raw,
                "search_512": policy_512,
                "search_800": policy_800,
                "consensus": consensus,
            }.items()
        },
        "effective_actions": {
            name: math.exp(_entropy(policy))
            for name, policy in {
                "raw": raw,
                "search_512": policy_512,
                "search_800": policy_800,
                "consensus": consensus,
            }.items()
        },
        "verified_expected_value": expected,
        "consensus_expected_delta_vs_raw": delta,
        "consensus_expected_delta_vs_800": expected["consensus"] - expected["search_800"],
        "consensus_top_action": consensus_top,
        "consensus_top_verified_delta_vs_raw_top": values[consensus_top] - raw_top_value,
        "improvement_probability_mass": improvement_mass,
        "harmful_probability_mass": harmful_mass,
        "policies": {
            "raw": tuple(sorted(raw.items())),
            "search_800": tuple(sorted(policy_800.items())),
            "consensus": tuple(sorted(consensus.items())),
        },
    }


def _summary(
    rows: tuple[Mapping[str, object], ...], *, config: ConsensusTargetConfig, seed: int
) -> dict[str, object]:
    qualified = tuple(row for row in rows if row["qualified"])
    deltas = tuple(float(row["consensus_expected_delta_vs_raw"]) for row in rows)
    qualified_deltas = tuple(
        float(row["consensus_expected_delta_vs_raw"]) for row in qualified
    )
    deltas_vs_800 = tuple(float(row["consensus_expected_delta_vs_800"]) for row in rows)
    harmful = sum(delta <= config.maximum_action_harm for delta in deltas)
    return {
        "positions": len(rows),
        "qualified_count": len(qualified),
        "qualified_ratio": len(qualified) / len(rows),
        "harmful_row_count": harmful,
        "harmful_row_ratio": harmful / len(rows),
        "mean_anchor_to_consensus_tv": mean(
            float(row["anchor_to_consensus_tv"]) for row in rows
        ),
        "mean_anchor_to_consensus_jsd": mean(
            float(row["anchor_to_consensus_jsd"]) for row in rows
        ),
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


def _gate(summary: Mapping[str, object], config: ConsensusTargetConfig) -> dict[str, object]:
    reasons = []
    if float(summary["qualified_ratio"]) < config.minimum_qualified_ratio:
        reasons.append("qualified target ratio is below 20%")
    if float(summary["harmful_row_ratio"]) > config.maximum_harmful_row_ratio:
        reasons.append("harmful target row ratio exceeds 10%")
    if float(summary["expected_delta_vs_raw_95_interval"][0]) <= 0:
        reasons.append("full consensus expected-value interval is not positive")
    if float(summary["qualified_expected_delta_95_interval"][0]) <= 0:
        reasons.append("qualified consensus expected-value interval is not positive")
    if float(summary["mean_anchor_to_consensus_tv"]) > config.maximum_mean_anchor_tv:
        reasons.append("mean anchor-to-consensus TV exceeds 0.125")
    if float(summary["mean_expected_delta_vs_800"]) < -config.maximum_value_regression_vs_800:
        reasons.append("consensus expected value regresses 800-search by more than 0.01")
    return {"passed": not reasons, "reasons": reasons}


def _record_index(records: tuple[ReplayRecord, ...]) -> dict[tuple[str, int, int], ReplayRecord]:
    return {(record.game_id, record.game_index, record.ply): record for record in records}


def run_consensus_target_audit(config: ConsensusTargetConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"consensus target output already exists: {config.output_dir}")
    source = json.loads(config.consistency_result.read_text(encoding="utf-8"))
    if source.get("gate", {}).get("passed"):
        raise ValueError("consensus audit expects the failed MIHENK consistency artifact")
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
                actions = {
                    action_to_legal_move(board, action).uci() for action, _ in record.raw_policy
                }
                for action in sorted(actions):
                    child = rules.apply(record.state, ChessMove(action))
                    work.append((len(matched) - 1, action, child))

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
    result_path = config.output_dir / "consensus.json"
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
    path = run_consensus_target_audit(
        ConsensusTargetConfig(
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
