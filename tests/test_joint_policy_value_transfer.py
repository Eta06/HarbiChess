from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.core.state import Side  # noqa: E402
from harbichess.replay.schema import ReplayRecord  # noqa: E402
from harbichess.training.joint_policy_value_transfer import (  # noqa: E402
    JointPolicyValueTransferConfig,
    OutcomeGameBalancedSampler,
    _audit_perspective,
    _spearman,
    _value_gate_reasons,
    _value_quality,
)


def _record(game: str, ply: int, side: Side, outcome: int | None) -> ReplayRecord:
    return ReplayRecord(
        game_id=game,
        game_index=0,
        seed=1,
        ply=ply,
        root_fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        moves=tuple("a1a2" for _ in range(ply)),
        side_to_move=side,
        policy=((0, 1.0),),
        selected_action=0,
        root_value=0.0,
        outcome_value=outcome,
        repetition_redirected=False,
    )


def test_outcome_sampler_uses_every_class_and_rejects_unknown_rows() -> None:
    records = tuple(
        _record(f"game-{outcome}-{index}", 0, Side.WHITE, outcome)
        for outcome in (-1, 0, 1)
        for index in range(2)
    )
    sampler = OutcomeGameBalancedSampler(records, seed=7)

    selected = sampler.sample_indices(12)

    assert {records[index].outcome_value for index in selected} == {-1, 0, 1}
    assert sum(records[index].outcome_value == -1 for index in selected) == 4
    assert sum(records[index].outcome_value == 0 for index in selected) == 4
    assert sum(records[index].outcome_value == 1 for index in selected) == 4
    with pytest.raises(ValueError, match="known outcomes"):
        OutcomeGameBalancedSampler((*records, _record("unknown", 0, Side.WHITE, None)), seed=7)


def test_perspective_audit_accepts_alternating_decisive_and_unknown_games() -> None:
    records = (
        _record("decisive", 0, Side.WHITE, 1),
        _record("decisive", 1, Side.BLACK, -1),
        _record("draw", 0, Side.WHITE, 0),
        _record("unknown", 0, Side.WHITE, None),
    )

    assert _audit_perspective(records) == {
        "games": 3,
        "decisive": 1,
        "draw": 1,
        "unknown": 1,
    }
    with pytest.raises(ValueError, match="perspective"):
        _audit_perspective(
            (
                _record("broken", 0, Side.WHITE, 1),
                _record("broken", 1, Side.BLACK, 1),
            )
        )


def test_value_metrics_gate_requires_calibrated_outcome_separation() -> None:
    outcomes = (-1, 0, 1) * 8
    baseline = _value_quality(mx.zeros((24, 3)), outcomes)
    candidate_rows = tuple(
        (0.0, 0.0, 3.0) if outcome == -1 else (0.0, 3.0, 0.0) if outcome == 0 else (3.0, 0.0, 0.0)
        for outcome in outcomes
    )
    candidate = _value_quality(mx.array(candidate_rows), outcomes)

    assert _value_gate_reasons(baseline, candidate) == ()
    assert candidate["expected_value_by_outcome"]["1"] > 0.8
    assert candidate["expected_value_by_outcome"]["-1"] < -0.8


def test_spearman_preserves_order_and_detects_reversal() -> None:
    assert _spearman((1.0, 2.0, 3.0), (10.0, 20.0, 30.0)) == pytest.approx(1.0)
    assert _spearman((1.0, 2.0, 3.0), (30.0, 20.0, 10.0)) == pytest.approx(-1.0)


def test_joint_transfer_config_requires_aligned_schedules() -> None:
    with pytest.raises(ValueError, match="align"):
        JointPolicyValueTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            target_result=Path("target"),
            train_shard=Path("train"),
            validation_shard=Path("validation"),
            warmup_steps=21,
            validation_interval=20,
        )


def test_shared_representation_audit_is_explicit_opt_in() -> None:
    inputs = {
        "output_dir": Path("output"),
        "model_path": Path("model"),
        "target_result": Path("target"),
        "train_shard": Path("train"),
        "validation_shard": Path("validation"),
    }

    assert JointPolicyValueTransferConfig(**inputs).require_head_warmup_gate is True
    assert (
        JointPolicyValueTransferConfig(
            **inputs, require_head_warmup_gate=False
        ).require_head_warmup_gate
        is False
    )


def test_joint_transfer_requires_positive_loss_weights() -> None:
    inputs = {
        "output_dir": Path("output"),
        "model_path": Path("model"),
        "target_result": Path("target"),
        "train_shard": Path("train"),
        "validation_shard": Path("validation"),
    }

    balanced = JointPolicyValueTransferConfig(**inputs, policy_weight=0.25)
    assert balanced.policy_weight == 0.25
    assert balanced.value_weight == 1.0
    with pytest.raises(ValueError, match="optimizer"):
        JointPolicyValueTransferConfig(**inputs, policy_weight=0.0)
