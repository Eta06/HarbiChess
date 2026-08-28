from pathlib import Path

from harbichess.evaluation.action_value_dataset import (
    ActionValueDatasetConfig,
    _excluded_identities,
    _gate,
)


def _config() -> ActionValueDatasetConfig:
    return ActionValueDatasetConfig(
        excluded_q_result=Path("old.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_action_value_dataset_excludes_exact_prior_row_identities() -> None:
    payload = {
        "rows": {
            "train": [
                {"game_id": "g1", "game_index": 4, "ply": 12},
                {"game_id": "g2", "game_index": 5, "ply": 20},
            ]
        }
    }
    assert _excluded_identities(payload, "train") == {
        ("g1", 4, 12),
        ("g2", 5, 20),
    }


def test_action_value_dataset_gate_requires_stable_verified_q_labels() -> None:
    passing = {
        "mean_high_q_verified_spearman": 0.35,
        "mean_cross_budget_q_spearman": 0.70,
        "mean_cross_budget_q_drift": 0.03,
        "mean_top_two_q_overlap": 0.75,
        "top_q_verified_delta_95_interval": (0.001, 0.10),
        "top_q_harmful_ratio": 0.10,
        "mean_top_q_verified_regret": 0.10,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {
        **passing,
        "mean_cross_budget_q_drift": 0.031,
        "mean_top_two_q_overlap": 0.70,
    }
    assert _gate(failed, _config())["reasons"] == [
        "cross-budget Q drift exceeds 0.03",
        "top-two Q-set overlap is below 75%",
    ]
