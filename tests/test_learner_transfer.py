import json
from pathlib import Path

import pytest

from harbichess.evaluation.model_quality import ModelQualityMetrics
from harbichess.training.learner_transfer import (
    LearnerTransferConfig,
    _candidate_reasons,
    _config_payload,
    _select_validation_snapshots,
    _validate_replay_alignment,
)


def _quality(
    *, policy: float, value: float, ece: float, top_agreement: float = 0.5
) -> ModelQualityMetrics:
    return ModelQualityMetrics(
        samples=100,
        known_value_samples=80,
        teacher_policy_cross_entropy=policy,
        global_teacher_policy_cross_entropy=8.0,
        teacher_top_action_agreement=top_agreement,
        value_cross_entropy=value,
        value_accuracy=0.4,
        expected_score_ece=ece,
        expected_score_brier=0.2,
    )


def _tactical(raw: int, searched: tuple[int, int]) -> dict[str, object]:
    return {
        "raw": {"solved": raw},
        "budgets": tuple({"solved": solved} for solved in searched),
    }


def test_transfer_gate_requires_policy_value_calibration_and_tactical_retention() -> None:
    config = LearnerTransferConfig(
        replay_run_result=Path("replay.json"),
        teacher_audit_result=Path("teacher.json"),
        output_dir=Path("output"),
    )
    baseline = _quality(policy=4.0, value=1.0, ece=0.10)
    baseline_tactical = _tactical(4, (6, 8))

    passed = _candidate_reasons(
        _quality(policy=3.8, value=1.01, ece=0.11),
        _tactical(4, (6, 8)),
        baseline_quality=baseline,
        baseline_tactical=baseline_tactical,
        config=config,
        maximum_gradient_norm=1.0,
    )
    failed = _candidate_reasons(
        _quality(policy=3.95, value=1.03, ece=0.13, top_agreement=0.4),
        _tactical(3, (5, 7)),
        baseline_quality=baseline,
        baseline_tactical=baseline_tactical,
        config=config,
        maximum_gradient_norm=6.0,
    )

    assert not passed
    assert len(failed) == 7
    assert "teacher top-action agreement regressed" in failed


def test_transfer_checkpoint_selection_is_even_and_metric_blind() -> None:
    snapshots = [(step, float(100 - step), object()) for step in range(10)]

    selected = _select_validation_snapshots(snapshots, maximum=4)

    assert [step for step, _, _ in selected] == [0, 3, 6, 9]


def test_transfer_requires_matching_passed_replay_alignment() -> None:
    replay = Path("artifacts/runs/fresh/result.json")

    _validate_replay_alignment(
        {"gate": {"passed": True}, "config": {"run_result": str(replay)}},
        replay_run_result=replay,
    )

    with pytest.raises(ValueError, match="passed fresh replay alignment"):
        _validate_replay_alignment(
            {"gate": {"passed": False}, "config": {"run_result": str(replay)}},
            replay_run_result=replay,
        )


def test_transfer_config_payload_serializes_alignment_path() -> None:
    config = LearnerTransferConfig(
        replay_run_result=Path("replay.json"),
        teacher_audit_result=Path("teacher.json"),
        replay_alignment_result=Path("alignment.json"),
        output_dir=Path("output"),
    )

    payload = _config_payload(config)

    assert payload["replay_alignment_result"] == "alignment.json"
    json.dumps(payload)
