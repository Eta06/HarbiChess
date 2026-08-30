import mlx.core as mx

from harbichess.training.constrained_plastic_ablation import (
    _combine_gradients,
    _gradient_statistics,
)


def _scalar(tree) -> float:
    value = dict(tree)["weight"]
    mx.eval(value)
    return float(value.item())


def test_gradient_statistics_expose_destructive_interference() -> None:
    statistics = _gradient_statistics({"weight": mx.array([1.0])}, {"weight": mx.array([-2.0])})

    assert statistics == {
        "dot": -2.0,
        "historical_norm": 1.0,
        "fresh_norm": 2.0,
        "cosine": -1.0,
    }


def test_mean_control_averages_domain_gradients() -> None:
    combined, weights = _combine_gradients(
        {"weight": mx.array([2.0])},
        {"weight": mx.array([4.0])},
        mode="mean-control",
    )

    assert _scalar(combined) == 3.0
    assert weights == {"historical": 0.5, "fresh": 0.5}


def test_pcgrad_removes_opposing_components() -> None:
    combined, _ = _combine_gradients(
        {"weight": mx.array([1.0, 0.0])},
        {"weight": mx.array([-1.0, 1.0])},
        mode="pcgrad",
    )
    values = dict(combined)["weight"]
    mx.eval(values)

    assert values.tolist() == [0.25, 0.75]


def test_mgda_selects_minimum_norm_convex_combination() -> None:
    combined, weights = _combine_gradients(
        {"weight": mx.array([1.0, 0.0])},
        {"weight": mx.array([0.0, 1.0])},
        mode="mgda",
    )
    values = dict(combined)["weight"]
    mx.eval(values)

    assert weights == {"historical": 0.5, "fresh": 0.5}
    assert values.tolist() == [0.5, 0.5]
