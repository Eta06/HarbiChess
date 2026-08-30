"""Qualify continuation ranking and Full Gumbel retention for a frozen value head."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.deterministic_value_probe import _prepare
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    select_stratified_records,
)
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.training.full_gumbel_transfer import _evaluator, _network
from harbichess.training.joint_policy_value_transfer import (
    _continuation_ranking,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class DecoupledValueQualificationConfig:
    output_dir: Path
    value_result: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    expected_candidate_sha256: str = (
        "6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb"
    )
    ranking_positions: int = 32
    ranking_depth: int = 4
    ranking_seed: int = 2026083091
    tactical_seed: int = 2026082883
    search_workers: int = 24
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025

    def __post_init__(self) -> None:
        if (
            min(
                self.ranking_positions,
                self.ranking_depth,
                self.ranking_seed,
                self.tactical_seed,
                self.search_workers,
                self.fixed_inference_batch_size,
            )
            <= 0
            or self.inference_wait_seconds < 0
        ):
            raise ValueError("decoupled value qualification configuration is invalid")
        if len(self.expected_candidate_sha256) != 64:
            raise ValueError("expected candidate hash must be SHA-256")


def _tactical(network, *, config: DecoupledValueQualificationConfig) -> dict[str, object]:
    rules = PythonChessRules()
    batcher, evaluator = _evaluator(
        network,
        config.search_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    try:
        return run_tactical_sweep(
            evaluator,
            rules=rules,
            budgets=(256,),
            workers=8,
            seed=config.tactical_seed,
            search_kind="full-gumbel",
            max_considered_actions=16,
            gumbel_scale=0.0,
        )
    finally:
        batcher.close()


def _tactical_gate(baseline: dict[str, object], candidate: dict[str, object]) -> tuple[str, ...]:
    baseline_cases = {
        row["case"]
        for row in baseline["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    candidate_cases = {
        row["case"]
        for row in candidate["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    reasons = []
    if int(candidate["raw"]["solved"]) < int(baseline["raw"]["solved"]):  # type: ignore[index]
        reasons.append("raw tactical solve count regressed")
    if int(candidate["budgets"][0]["solved"]) < 4:  # type: ignore[index]
        reasons.append("256 Full Gumbel tactical solve count is below four")
    if baseline_cases - candidate_cases:
        reasons.append("candidate search lost a baseline-solved tactical case")
    return tuple(reasons)


def run_decoupled_value_qualification(config: DecoupledValueQualificationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"value qualification output exists: {config.output_dir}")
    source = json.loads(config.value_result.read_text(encoding="utf-8"))
    selected_arm = source.get("selected_wdl_arm")
    if not source.get("passed") or selected_arm != "global-wdl":
        raise ValueError("qualification requires the frozen selected global WDL arm")
    candidate_path = Path(source["wdl_arms"][selected_arm]["model_path"])
    candidate_sha256 = _sha256(candidate_path)
    if candidate_sha256 != config.expected_candidate_sha256:
        raise ValueError("frozen candidate hash does not match preregistration")

    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    _, validation_records, split = _split_games(games, seed=pool_config.seed)
    rules = PythonChessRules()
    ranking_records = select_stratified_records(
        validation_records,
        rules=rules,
        count=config.ranking_positions,
        seed=config.ranking_seed,
    )
    validation_inputs, _ = _prepare(validation_records, rules)

    baseline = _network()
    baseline.load_weights(str(config.model_path))
    candidate = HarbiChessDecoupledValueNetwork.from_base(_network())
    candidate.load_weights(str(candidate_path))
    baseline_policy = baseline(validation_inputs)[0]
    candidate_policy = candidate(validation_inputs)[0]
    mx.eval(baseline_policy, candidate_policy)
    policy_max_abs_delta = float(mx.max(mx.abs(baseline_policy - candidate_policy)).item())

    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.EVALUATION,
        mode_detail="MIHVER continuation action-value qualification",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=2,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    continuation = _continuation_ranking(
        baseline,
        candidate,
        ranking_records,
        rules=rules,
        depth=config.ranking_depth,
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode_detail="MIHVER Full Gumbel tactical retention",
        pilot_steps_completed=1,
    )
    store.write_atomic(snapshot)
    baseline_tactical = _tactical(baseline, config=config)
    candidate_tactical = _tactical(candidate, config=config)
    tactical_reasons = _tactical_gate(baseline_tactical, candidate_tactical)
    reasons = [*continuation["reasons"], *tactical_reasons]
    if policy_max_abs_delta != 0.0:
        reasons.append("candidate policy logits changed")
    passed = not reasons
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
                        "runs_root",
                        "telemetry_path",
                    )
                },
            },
            "provenance": provenance,
            "split": split,
            "candidate": {"path": str(candidate_path), "sha256": candidate_sha256},
            "policy_max_abs_delta": policy_max_abs_delta,
            "continuation": continuation,
            "tactical": {
                "baseline": baseline_tactical,
                "candidate": candidate_tactical,
                "passed": not tactical_reasons,
                "reasons": tactical_reasons,
            },
            "passed": passed,
            "reasons": reasons,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        "MIHVER value representation qualified · continuous review required"
        if passed
        else "MIHVER downstream value gate failed · continuous blocked"
    )
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=detail,
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            pilot_steps_completed=2,
            pilot_stop_reason="value_downstream_gate",
            pilot_stop_detail=detail,
            pilot_reasons=tuple(reasons),
            promotion_ready=False,
        )
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    print(
        run_decoupled_value_qualification(
            DecoupledValueQualificationConfig(
                output_dir=arguments.output_dir,
                value_result=arguments.value_result,
                model_path=arguments.model,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
