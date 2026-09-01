"""Run the preregistered DENGE scalar value-calibration diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.plastic_value_network import HarbiChessPlasticValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.dashboard.state import CheckpointStatus, PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.decoupled_value_qualification import (
    DecoupledValueQualificationConfig,
    _tactical,
)
from harbichess.evaluation.deterministic_value_probe import _prepare
from harbichess.evaluation.teacher_qualification import _atomic_json, select_stratified_records
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.training.full_gumbel_transfer import _network
from harbichess.training.invariant_wdl_transfer import _wdl_quality
from harbichess.training.joint_policy_value_transfer import (
    _continuation_ranking,
    _sha256,
)
from harbichess.training.value_calibration import (
    fit_guarded_scalar_calibration,
    fit_scalar_calibration,
)


@dataclass(frozen=True, slots=True)
class ValueCalibrationAblationConfig:
    output_dir: Path
    source_result: Path
    value_result: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    split_seed: int = 2026091601
    ranking_seed: int = 2026095001
    ranking_positions: int = 1_440
    ranking_depth: int = 4
    tactical_seed: int = 2026082883
    search_workers: int = 24
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025
    guard_pearson_margin: float | None = None
    old_split_seed: int = 2026091603

    def __post_init__(self) -> None:
        if min(
            self.split_seed,
            self.ranking_seed,
            self.ranking_positions,
            self.ranking_depth,
            self.tactical_seed,
            self.search_workers,
            self.fixed_inference_batch_size,
            self.old_split_seed,
        ) <= 0 or self.inference_wait_seconds < 0:
            raise ValueError("DENGE calibration ablation configuration is invalid")
        if self.guard_pearson_margin is not None and self.guard_pearson_margin < 0:
            raise ValueError("DENGE guard Pearson margin cannot be negative")


def _phase(record: ReplayRecord) -> str:
    if record.ply < 20:
        return "opening"
    if record.ply < 80:
        return "middlegame"
    return "endgame"


def _white_outcome(record: ReplayRecord) -> int:
    return int(record.outcome_value) * (1 if record.side_to_move is Side.WHITE else -1)


def _game_disjoint_halves(
    records: tuple[ReplayRecord, ...], *, seed: int
) -> tuple[tuple[ReplayRecord, ...], tuple[ReplayRecord, ...], dict[str, object]]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        if record.outcome_value is not None:
            by_game[record.game_id].append(record)
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for game_id, rows in by_game.items():
        first = min(rows, key=lambda row: row.ply)
        strata[(_phase(first), _white_outcome(first))].append(game_id)

    def key(game_id: str) -> bytes:
        return hashlib.blake2b(f"{seed}:{game_id}".encode(), digest_size=16).digest()

    fit_games: set[str] = set()
    for game_ids in strata.values():
        ordered = sorted(game_ids, key=key)
        fit_games.update(ordered[: len(ordered) // 2])
    test_games = set(by_game) - fit_games
    fit = tuple(record for record in records if record.game_id in fit_games)
    test = tuple(record for record in records if record.game_id in test_games)
    if not fit or not test or fit_games & test_games:
        raise ValueError("DENGE game split is empty or overlapping")
    return fit, test, {
        "fit_games": len(fit_games),
        "test_games": len(test_games),
        "fit_rows": len(fit),
        "test_rows": len(test),
        "game_overlap": 0,
        "strata": {
            f"{phase}:{outcome}": len(game_ids)
            for (phase, outcome), game_ids in sorted(strata.items())
        },
    }


def _load_networks(config: ValueCalibrationAblationConfig, source: dict[str, object]):
    value = json.loads(config.value_result.read_text(encoding="utf-8"))
    selected = value.get("selected_wdl_arm")
    if not value.get("passed") or selected != "global-wdl":
        raise ValueError("DENGE requires the qualified MIHVER global WDL arm")
    mihver = HarbiChessDecoupledValueNetwork.from_base(_network())
    mihver.load_weights(value["wdl_arms"][selected]["model_path"])
    raw = HarbiChessPlasticValueNetwork.from_mihver(mihver)
    checkpoint = Path(source["updates"][-1]["checkpoint_path"])
    if _sha256(checkpoint) != source["updates"][-1]["checkpoint_sha256"]:
        raise ValueError("DENGE source checkpoint hash does not match its result")
    raw.load_weights(str(checkpoint), strict=False)
    calibrated = HarbiChessPlasticValueNetwork.from_mihver(mihver)
    calibrated.load_weights(list(tree_flatten(raw.parameters())))
    mx.eval(raw.parameters(), calibrated.parameters())
    return raw, calibrated, checkpoint


def _labels(records: tuple[ReplayRecord, ...]) -> tuple[int, ...]:
    return tuple({1: 0, 0: 1, -1: 2}[int(record.outcome_value)] for record in records)


def _metric_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, float]:
    return {
        metric: float(after[metric]) - float(before[metric])
        for metric in ("cross_entropy", "macro_cross_entropy", "brier", "ece_10")
    } | {
        "expected_score_pearson": float(after["expected_score_pearson"])
        - float(before["expected_score_pearson"])
    }


def _tactical_reasons(before: dict[str, object], after: dict[str, object]) -> tuple[str, ...]:
    before_cases = {
        row["case"]
        for row in before["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    after_cases = {
        row["case"]
        for row in after["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    reasons = []
    if int(after["budgets"][0]["solved"]) < 5:  # type: ignore[index]
        reasons.append("calibrated Full Gumbel tactical solve count is below 5/8")
    if before_cases - after_cases:
        reasons.append("calibration lost a PUSULA-16 solved tactical case")
    return tuple(reasons)


def run_value_calibration_ablation(config: ValueCalibrationAblationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"DENGE output exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    source = json.loads(config.source_result.read_text(encoding="utf-8"))
    if source.get("passed") or source.get("reasons") != [
        "cumulative statistical gate failed: fresh_ece_noninferior"
    ]:
        raise ValueError("DENGE requires the frozen failed PUSULA-16 calibration result")
    raw, calibrated, checkpoint = _load_networks(config, source)
    rules = PythonChessRules()
    store = SnapshotStore(config.telemetry_path)

    def publish(detail: str) -> None:
        store.write_atomic(
            replace(
                store.read(),
                updated_at=datetime.now(UTC).isoformat(),
                mode=RunMode.EVALUATION,
                mode_detail=detail,
                run_id=config.output_dir.name,
                pilot_status=PilotStatus.TRAINING,
                promotion_ready=False,
            )
        )

    publish("DENGE calibration ablation · fit/test logits")
    fresh_records = tuple(
        record
        for record in read_shard(Path(source["final_qualification"]["path"])).records
        if record.outcome_value is not None
    )
    fit_records, test_records, split = _game_disjoint_halves(
        fresh_records, seed=config.split_seed
    )
    fit_inputs, _ = _prepare(fit_records, rules)
    fit_logits = raw(fit_inputs)[1]
    old_records = tuple(
        record
        for record in read_shard(Path(source["old_qualification"]["path"])).records
        if record.outcome_value is not None
    )
    old_split = None
    if config.guard_pearson_margin is None:
        calibration = fit_scalar_calibration(
            fit_logits,
            _labels(fit_records),
            group_ids=tuple(record.game_id for record in fit_records),
        )
        selected_calibration = calibration
        old_test_records = old_records
    else:
        old_guard_records, old_test_records, old_split = _game_disjoint_halves(
            old_records, seed=config.old_split_seed
        )
        old_guard_inputs, _ = _prepare(old_guard_records, rules)
        old_guard_logits = raw(old_guard_inputs)[1]
        calibration = fit_guarded_scalar_calibration(
            fit_logits,
            _labels(fit_records),
            old_guard_logits,
            tuple(int(record.outcome_value) for record in old_guard_records),
            fit_group_ids=tuple(record.game_id for record in fit_records),
            guard_pearson_margin=config.guard_pearson_margin,
        )
        selected_calibration = calibration.selected
    calibrated.set_value_logit_scale(selected_calibration.logit_scale)
    test_inputs, _ = _prepare(test_records, rules)
    raw_test = _wdl_quality(raw, test_inputs, tuple(int(r.outcome_value) for r in test_records))
    calibrated_test = _wdl_quality(
        calibrated, test_inputs, tuple(int(r.outcome_value) for r in test_records)
    )
    policy_inputs = mx.take(test_inputs, mx.arange(min(256, test_inputs.shape[0])), axis=0)
    raw_policy = raw(policy_inputs)[0]
    calibrated_policy = calibrated(policy_inputs)[0]
    mx.eval(raw_policy, calibrated_policy)
    policy_bitwise = bool(mx.array_equal(raw_policy, calibrated_policy).item())

    publish("DENGE calibration ablation · old capability")
    old_inputs, _ = _prepare(old_test_records, rules)
    old_outcomes = tuple(int(record.outcome_value) for record in old_test_records)
    raw_old = _wdl_quality(raw, old_inputs, old_outcomes)
    calibrated_old = _wdl_quality(calibrated, old_inputs, old_outcomes)

    tactical_config = replace(
        DecoupledValueQualificationConfig(
            output_dir=config.output_dir,
            value_result=config.value_result,
            model_path=config.model_path,
        ),
        tactical_seed=config.tactical_seed,
        search_workers=config.search_workers,
        fixed_inference_batch_size=config.fixed_inference_batch_size,
        inference_wait_seconds=config.inference_wait_seconds,
    )
    publish("DENGE calibration ablation · Full Gumbel tactical")
    raw_tactical = _tactical(raw, config=tactical_config)
    calibrated_tactical = _tactical(calibrated, config=tactical_config)

    publish("DENGE calibration ablation · 1440-position continuation")
    games, _ = _load_games(
        CorrectedReplayValueTransferConfig(
            output_dir=config.output_dir,
            model_path=config.model_path,
            runs_root=config.runs_root,
        )
    )
    _, validation_records, _ = _split_games(
        games, seed=CorrectedReplayValueTransferConfig(
            output_dir=config.output_dir,
            model_path=config.model_path,
            runs_root=config.runs_root,
        ).seed
    )
    ranking_records = select_stratified_records(
        validation_records,
        rules=rules,
        count=config.ranking_positions,
        seed=config.ranking_seed,
    )
    release = _network()
    release.load_weights(str(config.model_path))
    raw_continuation = _continuation_ranking(
        release, raw, ranking_records, rules=rules, depth=config.ranking_depth
    )
    calibrated_continuation = _continuation_ranking(
        release, calibrated, ranking_records, rules=rules, depth=config.ranking_depth
    )

    test_delta = _metric_delta(raw_test, calibrated_test)
    old_delta = _metric_delta(raw_old, calibrated_old)
    reasons = []
    if test_delta["ece_10"] > -0.020:
        reasons.append("fresh test ECE improvement is below 0.020")
    if test_delta["cross_entropy"] > 0 or test_delta["brier"] > 0:
        reasons.append("fresh test CE or Brier regressed")
    if test_delta["expected_score_pearson"] < -0.005:
        reasons.append("fresh test Pearson regressed by more than 0.005")
    if old_delta["cross_entropy"] > 0.003 or old_delta["brier"] > 0.003:
        reasons.append("old diagnostic CE or Brier exceeded its 0.003 margin")
    if old_delta["expected_score_pearson"] < -0.010:
        reasons.append("old diagnostic Pearson exceeded its 0.010 margin")
    if float(calibrated_old["ece_10"]) > 0.120:
        reasons.append("old diagnostic absolute ECE exceeds 0.120")
    if not policy_bitwise:
        reasons.append("calibration changed policy logits")
    reasons.extend(_tactical_reasons(raw_tactical, calibrated_tactical))
    continuation_spearman_delta = (
        float(calibrated_continuation["candidate_mean_spearman"])
        - float(raw_continuation["candidate_mean_spearman"])
    )
    continuation_top_delta = (
        float(calibrated_continuation["candidate_verified_top_agreement"])
        - float(raw_continuation["candidate_verified_top_agreement"])
    )
    if continuation_spearman_delta < -0.020:
        reasons.append("continuation Spearman exceeded its 0.020 margin")
    if continuation_top_delta < -1.0 / config.ranking_positions:
        reasons.append("continuation verified-top lost more than one position")

    result = {
        "passed": not reasons,
        "diagnostic_only": True,
        "reasons": reasons,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_result": str(config.source_result),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "split": split,
        "calibration": calibration.to_dict(),
        "old_split": old_split,
        "fresh_test": {
            "raw": raw_test,
            "calibrated": calibrated_test,
            "delta_calibrated_minus_raw": test_delta,
        },
        "old_diagnostic": {
            "raw": raw_old,
            "calibrated": calibrated_old,
            "delta_calibrated_minus_raw": old_delta,
        },
        "policy_bitwise_equal": policy_bitwise,
        "tactical": {"raw": raw_tactical, "calibrated": calibrated_tactical},
        "continuation": {
            "positions": config.ranking_positions,
            "raw": raw_continuation,
            "calibrated": calibrated_continuation,
            "spearman_delta": continuation_spearman_delta,
            "verified_top_delta": continuation_top_delta,
        },
    }
    result_path = config.output_dir / "result.json"
    _atomic_json(result_path, result)
    store.write_atomic(
        replace(
            store.read(),
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=(
                "DENGE calibration diagnostic passed"
                if not reasons
                else "DENGE calibration diagnostic failed"
            ),
            pilot_status=PilotStatus.PASSED if not reasons else PilotStatus.FAILED,
            pilot_reasons=tuple(reasons),
            checkpoint_status=(
                CheckpointStatus.VERIFIED if not reasons else CheckpointStatus.FAILED
            ),
            checkpoint_verified=not reasons,
            promotion_ready=False,
        )
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--guard-pearson-margin", type=float)
    arguments = parser.parse_args(argv)
    result = run_value_calibration_ablation(
        ValueCalibrationAblationConfig(
            output_dir=arguments.output_dir,
            source_result=arguments.source_result,
            value_result=arguments.value_result,
            model_path=arguments.model,
            runs_root=arguments.runs_root,
            telemetry_path=arguments.telemetry,
            guard_pearson_margin=arguments.guard_pearson_margin,
        )
    )
    print(result)
    return 0 if json.loads(result.read_text(encoding="utf-8"))["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
