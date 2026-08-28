from pathlib import Path

from harbichess.training.anchored_policy_transfer import (
    AnchoredPolicyTransferConfig,
    _reasons,
    _split_games,
)


def _config(tmp_path: Path) -> AnchoredPolicyTransferConfig:
    return AnchoredPolicyTransferConfig(
        policy_target_result=tmp_path / "target.json",
        dataset_result=tmp_path / "dataset.json",
        run_result=tmp_path / "run.json",
        train_shard=tmp_path / "train.jsonl.gz",
        output_dir=tmp_path / "output",
    )


def test_game_split_is_deterministic_and_disjoint() -> None:
    rows = tuple(
        {"game_id": game_id, "ply": ply}
        for game_id in ("a", "b", "c", "d", "e")
        for ply in range(3)
    )

    fit, holdout = _split_games(rows, fraction=0.2, seed=17)

    assert fit.isdisjoint(holdout)
    assert fit | holdout == {"a", "b", "c", "d", "e"}
    assert len(holdout) == 1
    assert (fit, holdout) == _split_games(rows, fraction=0.2, seed=17)


def test_frozen_cipa_matrix_matches_preregistration(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.anchor_positions == 2_048
    assert config.rank == 32
    assert config.steps == 960
    assert config.target_batch_size == 16
    assert config.anchor_batch_size == 64
    assert config.anchor_weights == (0.25, 1.0, 4.0)
    assert config.arm_seeds == (2026082853, 2026082854, 2026082855)
    assert config.maximum_anchor_kl == 0.02


def test_internal_gate_accepts_only_all_passing_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    quality = {
        "mean_teacher_policy_spearman": 0.5,
        "verified_delta_95_interval": (0.01, 0.04),
        "harmful_ratio": 0.08,
        "mean_verified_regret": 0.08,
        "best_action_coverage_top_16": 0.9,
    }

    assert not _reasons(
        quality,
        gap_fraction=0.25,
        anchor_kl=0.01,
        maximum_gradient_norm=4.0,
        config=config,
    )

    assert _reasons(
        quality,
        gap_fraction=0.25,
        anchor_kl=0.021,
        maximum_gradient_norm=4.0,
        config=config,
    ) == ("broad replay anchor KL exceeds 0.02",)
