from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.full_gumbel_anchor_ablation import (  # noqa: E402
    PolicyAnchorAblationConfig,
    _arm_reasons,
    _baseline_kl,
)


def _quality(cross_entropy: float, agreement: float) -> dict[str, dict[str, float]]:
    return {
        "train": {
            "cross_entropy": cross_entropy,
            "teacher_kl": cross_entropy - 1.0,
            "top_action_agreement": agreement,
        },
        "validation": {
            "cross_entropy": cross_entropy,
            "teacher_kl": cross_entropy - 1.0,
            "top_action_agreement": agreement,
        },
    }


def _tactical(solved: int, cases: tuple[str, ...]) -> dict[str, object]:
    rows = tuple(
        {"case": name, "solved": name in cases}
        for name in ("mate-a", "mate-b", "defense-a", "defense-b")
    )
    return {"raw": {"solved": solved}, "budgets": ({"solved": solved, "cases": rows},)}


def test_baseline_kl_is_zero_for_equal_policy_and_positive_for_drift() -> None:
    baseline = mx.array([[2.0, 0.0, -1.0]])
    masks = mx.array([[True, True, False]])

    assert _baseline_kl(baseline, baseline, masks) == pytest.approx(0.0, abs=1e-7)
    assert _baseline_kl(baseline, mx.array([[0.0, 2.0, 8.0]]), masks) > 1.0


def test_arm_gate_preserves_search_tactics_and_frozen_wdl() -> None:
    baseline_tactical = _tactical(4, ("mate-a", "mate-b", "defense-a", "defense-b"))
    candidate_tactical = _tactical(4, ("mate-a", "mate-b", "defense-a", "defense-b"))
    reasons = _arm_reasons(
        _quality(3.0, 0.10),
        _quality(2.9, 0.13),
        baseline_tactical,
        candidate_tactical,
        policy_kl=0.05,
        wdl_max_logit_delta=0.0,
        wdl_max_metric_delta=0.0,
        frozen_hash_before="same",
        frozen_hash_after="same",
    )

    assert reasons == ()
    assert "candidate search lost a baseline-solved tactical case" in _arm_reasons(
        _quality(3.0, 0.10),
        _quality(2.9, 0.13),
        baseline_tactical,
        _tactical(3, ("mate-a", "mate-b", "defense-a")),
        policy_kl=0.05,
        wdl_max_logit_delta=0.0,
        wdl_max_metric_delta=0.0,
        frozen_hash_before="same",
        frozen_hash_after="same",
    )
    assert "frozen WDL output or calibration metrics changed" in _arm_reasons(
        _quality(3.0, 0.10),
        _quality(2.9, 0.13),
        baseline_tactical,
        candidate_tactical,
        policy_kl=0.05,
        wdl_max_logit_delta=1e-4,
        wdl_max_metric_delta=0.0,
        frozen_hash_before="same",
        frozen_hash_after="same",
    )


def test_anchor_config_rejects_duplicate_or_nonpositive_weights() -> None:
    common = {
        "output_dir": Path("output"),
        "model_path": Path("model"),
        "target_result": Path("target"),
        "train_shard": Path("train"),
        "validation_shard": Path("validation"),
    }
    with pytest.raises(ValueError, match="unique and positive"):
        PolicyAnchorAblationConfig(**common, anchor_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="unique and positive"):
        PolicyAnchorAblationConfig(**common, anchor_weights=(0.0,))
