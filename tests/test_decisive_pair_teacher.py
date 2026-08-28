from pathlib import Path

from harbichess.evaluation.decisive_pair_teacher import (
    DecisivePairTeacherConfig,
    _decisive_pair_score,
    _gate,
)


def _config() -> DecisivePairTeacherConfig:
    return DecisivePairTeacherConfig(
        dataset_result=Path("dataset.json"),
        label_result=Path("labels.json"),
        output_dir=Path("output"),
    )


def test_decisive_pairs_ignore_verifier_ties_and_score_order() -> None:
    score, count = _decisive_pair_score(
        {"a": 0.8, "b": 0.6, "c": 0.4},
        {"a": 0.9, "b": 0.88, "c": 0.1},
        minimum_margin=0.05,
    )
    assert count == 2
    assert score == 1.0


def test_decisive_pair_gate_retains_strength_and_safety() -> None:
    summary = {
        "informative_position_ratio": 0.50,
        "mean_decisive_pair_concordance": 0.60,
        "decisive_pair_concordance_95_interval": (0.51, 0.70),
        "labelable_ratio": 0.95,
        "mean_common_support_fraction": 0.95,
        "mean_stable_visit_mass": 0.80,
        "conservative_verified_delta_95_interval": (0.001, 0.05),
        "conservative_harmful_ratio": 0.10,
        "mean_conservative_verified_regret": 0.10,
    }
    assert _gate(summary, _config()) == {"passed": True, "reasons": []}

    failed = {**summary, "decisive_pair_concordance_95_interval": (0.50, 0.70)}
    assert _gate(failed, _config())["reasons"] == [
        "decisive-pair concordance interval lower bound is not above 0.50"
    ]
