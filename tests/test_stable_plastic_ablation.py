from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from harbichess.training.stable_plastic_ablation import (
    StablePlasticAblationConfig,
    _fresh_split,
    _strict_wdl_reasons,
)


def _config(**overrides) -> StablePlasticAblationConfig:
    values = {
        "output_dir": Path("output"),
        "value_result": Path("value.json"),
        "model_path": Path("model.safetensors"),
        "source_continuous_result": Path("continuous.json"),
    }
    values.update(overrides)
    return StablePlasticAblationConfig(**values)


def _wdl(**overrides) -> dict[str, float]:
    values = {
        "cross_entropy": 0.90,
        "macro_cross_entropy": 0.92,
        "expected_score_pearson": 0.45,
        "loss_draw_margin": 0.20,
        "win_draw_margin": 0.20,
        "ece_10": 0.03,
    }
    values.update(overrides)
    return values


def test_config_requires_batch_divisible_by_eight() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        _config(batch_size=66)


def test_strict_gate_rejects_any_cumulative_regression() -> None:
    assert _strict_wdl_reasons(_wdl(), _wdl(cross_entropy=0.89), label="old") == ()
    assert _strict_wdl_reasons(
        _wdl(),
        _wdl(cross_entropy=0.900001, expected_score_pearson=0.449999),
        label="old",
    ) == ("old cross_entropy regressed", "old expected-score Pearson regressed")


def test_fresh_split_is_deterministic_and_game_disjoint() -> None:
    records = tuple(
        SimpleNamespace(
            game_id=f"{kind}-{game}",
            outcome_value=(0 if kind == "draw" else (-1 if ply % 2 else 1)),
        )
        for kind in ("draw", "decisive")
        for game in range(8)
        for ply in range(4)
    )

    first = _fresh_split(records, seed=17)
    second = _fresh_split(records, seed=17)

    assert first == second
    train_games = {row.game_id for row in first[0]}
    validation_games = {row.game_id for row in first[1]}
    assert not train_games & validation_games
    assert first[2]["train_games"] == 12
    assert first[2]["validation_games"] == 4
    assert first[2]["overlap"] is False
