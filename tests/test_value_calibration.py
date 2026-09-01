import math

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.value_calibration import (  # noqa: E402
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
