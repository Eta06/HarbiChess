from pathlib import Path

import pytest

from harbichess.evaluation.cumulative_power_plan import (
    CumulativePowerPlanConfig,
    _game_ce_difference,
)
from harbichess.evaluation.cumulative_value_gate import PredictionGame


def _config(**overrides) -> CumulativePowerPlanConfig:
    values = {
        "output_dir": Path("output"),
        "value_result": Path("value.json"),
        "model_path": Path("model.safetensors"),
        "source_continuous_result": Path("continuous.json"),
        "pilot_candidate_path": Path("pilot.safetensors"),
    }
    values.update(overrides)
    return CumulativePowerPlanConfig(**values)


def test_power_config_rejects_design_effect_below_required_improvement() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        _config(fresh_ce_design_improvement=0.002)


def test_game_ce_difference_uses_paired_loss_direction() -> None:
    game = PredictionGame(
        game_id="game",
        outcomes=(1,),
        baseline_probabilities=((0.5, 0.25, 0.25),),
        candidate_probabilities=((0.75, 0.125, 0.125),),
    )

    improvement = _game_ce_difference(game, improvement=True)
    deterioration = _game_ce_difference(game, improvement=False)

    assert improvement > 0.0
    assert deterioration == -improvement


def test_power_config_requires_rounded_bounds() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        _config(minimum_games=193)
