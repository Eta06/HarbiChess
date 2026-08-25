"""Blend original and safe continuation policies by conservative value regret."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from harbichess.chess.rules import PythonChessRules
from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.repetition_risk import _atomic_json, _now, _source_commit
from harbichess.replay.schema import (
    PolicyRegretAdjustment,
    RepetitionRiskEstimate,
    ReplayRecord,
)
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit


@dataclass(frozen=True, slots=True)
class ValueRegretConfig:
    input_shard: Path
    risk_audit: Path
    original_shards: tuple[Path, ...]
    output_dir: Path
    temperature: float = 0.02

    def __post_init__(self) -> None:
        if not self.original_shards or not math.isfinite(self.temperature):
            raise ValueError("value regret requires original replay and finite temperature")
        if self.temperature <= 0.0:
            raise ValueError("value regret temperature must be positive")


@dataclass(frozen=True, slots=True)
class RootRegretAudit:
    game_id: str
    root_value: float
    best_nonrepeat_value: float
    regret: float
    redirect_fraction: float
    original_repeat_mass: float
    adjusted_repeat_mass: float
    redirect_actions: tuple[int, ...]


def blend_policy_by_regret(
    original: ReplayRecord,
    evidence_record: ReplayRecord,
    risks: tuple[RepetitionRiskEstimate, ...],
    *,
    temperature: float,
) -> ReplayRecord:
    evidence = evidence_record.continuation_evidence
    if evidence is None:
        raise ValueError("value regret requires v4 branch evidence")
    risks_by_action = {risk.action: risk for risk in risks}
    if set(risks_by_action) != set(evidence.qualified_actions):
        raise ValueError("value regret risks must cover every qualified v4 branch")
    redirect_values = {
        action: risks_by_action[action].risk_adjusted_value_lower_bound
        for action in evidence.qualified_actions
        if risks_by_action[action].risk_adjusted_value_lower_bound is not None
        and risks_by_action[action].risk_adjusted_value_lower_bound
        > evidence.repeat_value + evidence.minimum_advantage
    }
    if not redirect_values:
        return original
    best_nonrepeat = max(redirect_values.values())
    regret = max(
        0.0,
        min(evidence_record.root_value, best_nonrepeat) - evidence.repeat_value,
    )
    redirect_fraction = 1.0 - math.exp(-regret / temperature)
    redirect_total = sum(
        value - evidence.repeat_value for value in redirect_values.values()
    )
    redirect_policy = {
        action: (value - evidence.repeat_value) / redirect_total
        for action, value in redirect_values.items()
    }
    original_policy = dict(original.policy)
    blended = {
        action: (1.0 - redirect_fraction) * original_policy.get(action, 0.0)
        + redirect_fraction * redirect_policy.get(action, 0.0)
        for action in original_policy.keys() | redirect_policy.keys()
    }
    blended = {action: probability for action, probability in blended.items() if probability > 0}
    total = sum(blended.values())
    policy = tuple(sorted((action, probability / total) for action, probability in blended.items()))
    adjustment = PolicyRegretAdjustment(
        method_version=1,
        temperature=temperature,
        root_value=evidence_record.root_value,
        repeat_value=evidence.repeat_value,
        best_nonrepeat_value=best_nonrepeat,
        regret=regret,
        redirect_fraction=redirect_fraction,
        repeat_actions=evidence.repeat_actions,
        redirect_actions=tuple(sorted(redirect_values)),
        source_model_sha256=evidence.source_model_sha256,
    )
    return replace(
        original,
        policy=policy,
        repetition_redirected=original.repetition_redirected or redirect_fraction > 0.0,
        continuation_evidence=None,
        policy_regret_adjustment=adjustment,
    )


def run_value_regret(config: ValueRegretConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"value regret output already exists: {config.output_dir}")
    rules = PythonChessRules()
    evidence_shard = read_shard(config.input_shard, rules=rules)
    if evidence_shard.header.target_schema != 4:
        raise ValueError("value regret requires the immutable v4 evidence shard")
    originals = tuple(
        (path, read_shard(path, rules=rules)) for path in config.original_shards
    )
    original_by_game = {
        record.game_id: record
        for record in merge_continuation_replay(originals, recency_decay=1.0).records
    }
    risk_payload = json.loads(config.risk_audit.read_text(encoding="utf-8"))
    if risk_payload["output_shard"]["header"]["target_schema"] != 6:
        raise ValueError("value regret requires a corrected v6 risk audit")
    risks_by_game = {
        root["game_id"]: tuple(
            RepetitionRiskEstimate(**risk) for risk in root["repetition_risks"]
        )
        for root in risk_payload["roots"]
    }
    if any(
        record.game_id not in original_by_game or record.game_id not in risks_by_game
        for record in evidence_shard.records
    ):
        raise ValueError("value regret inputs do not cover every evidence root")

    targets = []
    audits = []
    for evidence_record in evidence_shard.records:
        original = original_by_game[evidence_record.game_id]
        target = blend_policy_by_regret(
            original,
            evidence_record,
            risks_by_game[evidence_record.game_id],
            temperature=config.temperature,
        )
        target.validate_rules(rules)
        targets.append(target)
        adjustment = target.policy_regret_adjustment
        if adjustment is None:
            continue
        original_policy = dict(original.policy)
        adjusted_policy = dict(target.policy)
        audits.append(
            RootRegretAudit(
                game_id=target.game_id,
                root_value=adjustment.root_value,
                best_nonrepeat_value=adjustment.best_nonrepeat_value,
                regret=adjustment.regret,
                redirect_fraction=adjustment.redirect_fraction,
                original_repeat_mass=sum(
                    original_policy.get(action, 0.0) for action in adjustment.repeat_actions
                ),
                adjusted_repeat_mass=sum(
                    adjusted_policy.get(action, 0.0) for action in adjustment.repeat_actions
                ),
                redirect_actions=adjustment.redirect_actions,
            )
        )

    config.output_dir.mkdir(parents=True)
    shard_path = config.output_dir / "continuation-value-regret.jsonl.gz"
    header = write_shard_atomic(
        shard_path,
        targets,
        ShardMetadata(
            run_id=config.output_dir.name,
            generation=evidence_shard.header.generation + 1,
            source_checkpoint=evidence_shard.header.source_checkpoint,
            source_commit=_source_commit(),
            created_at=_now(),
            split=ReplaySplit.TRAIN,
        ),
    )
    result_path = config.output_dir / "value-regret.json"
    _atomic_json(
        result_path,
        {
            "created_at": _now(),
            "source_commit": _source_commit(),
            "config": {
                "input_shard": str(config.input_shard),
                "risk_audit": str(config.risk_audit),
                "original_shards": [str(path) for path in config.original_shards],
                "output_dir": str(config.output_dir),
                "temperature": config.temperature,
            },
            "summary": {
                "roots": len(targets),
                "adjusted_roots": sum(item.redirect_fraction > 0 for item in audits),
                "defensive_roots": sum(item.redirect_fraction == 0 for item in audits),
                "mean_redirect_fraction": sum(item.redirect_fraction for item in audits)
                / len(audits),
                "maximum_redirect_fraction": max(item.redirect_fraction for item in audits),
                "original_repeat_mass": sum(item.original_repeat_mass for item in audits),
                "adjusted_repeat_mass": sum(item.adjusted_repeat_mass for item in audits),
            },
            "output_shard": {"path": str(shard_path), "header": asdict(header)},
            "roots": [asdict(item) for item in audits],
        },
    )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shard", required=True, type=Path)
    parser.add_argument("--risk-audit", required=True, type=Path)
    parser.add_argument("--original-shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_value_regret(
        ValueRegretConfig(
            input_shard=arguments.input_shard,
            risk_audit=arguments.risk_audit,
            original_shards=tuple(arguments.original_shard),
            output_dir=arguments.output_dir,
            temperature=arguments.temperature,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
