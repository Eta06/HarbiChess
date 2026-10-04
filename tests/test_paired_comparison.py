from copy import deepcopy

import pytest

from harbichess.evaluation.paired_comparison import compare


def games(scores, caps=()):
    return [
        {
            "opening_pair": i // 2,
            "opening": [f"root-{i // 2}"],
            "candidate_color": "white" if i % 2 == 0 else "black",
            "score": score,
            "termination": "max_plies" if i in caps else "CHECKMATE",
        }
        for i, score in enumerate(scores)
    ]


def test_matched_family_contrasts_and_conservative_unknown_outcomes():
    candidate = games([1, 1, 0.5, 0, 0.5, 0.5], caps=(4,))
    control = games([0, 0, 1, 0, 0.5, 0], caps=(4,))
    result = compare(candidate, control, replicates=2000)
    assert result["pair_values"] == [1, -0.25, 0.25]
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["score_bounds_unknown_caps"] == pytest.approx([1 / 6, 0.5])
    assert result["adjusted_alpha"] == pytest.approx(0.05 / 3)
    assert result["bootstrap_pair_adjusted"][0] <= result["bootstrap_pair_95"][0]
    assert result["bootstrap_pair_adjusted"][1] >= result["bootstrap_pair_95"][1]
    assert compare(candidate, control, replicates=2000) == result
    assert compare(candidate, candidate, replicates=100)["pair_values"] == [0, 0, 0]


def test_direct_scores_keep_unknown_caps_separate_from_bootstrap():
    result = compare(games([0.5] * 4, caps=range(4)), replicates=100)
    assert result["bootstrap_pair_adjusted"] == [0.5, 0.5]
    assert result["score_bounds_unknown_caps"] == [0, 1]
    assert result["mean"] == 0.5
    assert compare(games([1] * 4), replicates=100)["bootstrap_pair_adjusted"] == [1, 1]


@pytest.mark.parametrize("mutation", ["color", "root", "family", "score", "cap", "missing"])
def test_refuses_misalignment_and_invalid_results(mutation):
    candidate = games([1, 0, 0.5, 1])
    other = deepcopy(candidate)
    if mutation == "color":
        other[1]["candidate_color"] = "white"
    elif mutation == "root":
        other[0]["opening"] = other[1]["opening"] = ["different-root"]
    elif mutation == "family":
        other[2]["opening_pair"] = 0
    elif mutation == "score":
        other[2]["score"] = float("nan")
    elif mutation == "cap":
        other[0]["termination"] = "max_plies"
    else:
        other.pop()
    with pytest.raises(ValueError):
        compare(candidate, other, replicates=100)


def test_refuses_duplicate_source_roots_and_invalid_budgets():
    duplicate = games([1] * 4)
    duplicate[2]["opening"] = duplicate[3]["opening"] = duplicate[0]["opening"]
    with pytest.raises(ValueError):
        compare(duplicate, replicates=100)
    for kwargs in ({"replicates": 0}, {"alpha": 0}, {"alpha": 1}):
        with pytest.raises(ValueError):
            compare(games([1, 1]), **kwargs)
