import math

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.value_calibration import (  # noqa: E402
    fit_guarded_scalar_calibration,
    fit_scalar_calibration,
    scaled_logits,
)


def test_scalar_calibration_sharpens_underconfident_correct_logits() -> None:
    logits = mx.array(
        (
            (0.20, 0.00, -0.20),
            (-0.10, 0.20, -0.10),
            (-0.20, 0.00, 0.20),
        )
    )

    result = fit_scalar_calibration(logits, (0, 1, 2), maximum_scale=8.0)

    assert result.logit_scale > 1.0
    assert result.temperature < 1.0
    assert result.fit_cross_entropy_after < result.fit_cross_entropy_before
    assert scaled_logits(logits, result).shape == logits.shape


def test_scalar_calibration_balances_games_instead_of_long_rows() -> None:
    logits = mx.array(((1.0, 0.0, -1.0),) * 9 + ((1.0, 0.0, -1.0),))
    labels = (0,) * 9 + (2,)

    row_weighted = fit_scalar_calibration(logits, labels)
    game_weighted = fit_scalar_calibration(
        logits,
        labels,
        group_ids=("long",) * 9 + ("short",),
    )

    assert row_weighted.logit_scale > game_weighted.logit_scale
    assert game_weighted.logit_scale == pytest.approx(0.25)
    assert game_weighted.groups == 2


def test_guarded_calibration_clips_fresh_optimum_at_pearson_margin() -> None:
    fit_logits = mx.array(
        (
            (-0.176, -0.349, 0.151),
            (-0.428, 0.036, -0.134),
            (-0.442, 0.007, -0.463),
            (-0.066, -0.430, -0.409),
            (-0.075, 0.327, -0.376),
            (-0.277, 0.127, 0.448),
            (0.077, -0.103, 0.476),
            (-0.453, 0.358, -0.210),
        )
    )
    guard_logits = mx.array(
        (
            (-1.423, -1.529, -0.766),
            (-1.277, 0.326, 0.556),
            (-1.610, 0.848, 0.257),
            (-1.176, 0.722, -0.290),
            (-0.138, 1.694, -0.554),
            (1.178, 0.796, -1.024),
            (-0.799, -0.020, -0.626),
            (-0.848, 1.921, -1.528),
        )
    )

    result = fit_guarded_scalar_calibration(
        fit_logits,
        (2, 1, 1, 0, 1, 2, 2, 1),
        guard_logits,
        (1, 0, 1, 0, -1, 1, 0, 0),
        guard_pearson_margin=0.0,
        maximum_scale=4.0,
    )

    assert result.constraint_active
    assert 1.0 <= result.selected.logit_scale < result.unconstrained.logit_scale
    assert result.guard_pearson_selected >= result.guard_pearson_before - 1e-12


@pytest.mark.parametrize(
    ("logits", "labels", "message"),
    (
        (mx.array(((0.0, 0.0),)), (0,), "shape"),
        (mx.array(((0.0, 0.0, 0.0),)), (3,), "labels"),
        (mx.array(((math.nan, 0.0, 0.0),)), (0,), "finite"),
    ),
)
def test_scalar_calibration_validates_inputs(logits, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_scalar_calibration(logits, labels)
