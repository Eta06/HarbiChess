import pytest

from harbichess.selfplay.continuous_replay import ContinuousReplayConfig


def test_continuous_replay_defaults_match_frozen_pilot() -> None:
    config = ContinuousReplayConfig()

    assert config.games == 12
    assert config.simulations == 64
    assert config.gumbel_scale == 1.0
    assert config.exploration_plies == 30
    assert config.max_plies == 96


@pytest.mark.parametrize(
    ("name", "value"),
    (("games", 0), ("simulations", 0), ("gumbel_scale", 0.0), ("exploration_plies", -1)),
)
def test_continuous_replay_rejects_invalid_compute(name: str, value: int | float) -> None:
    with pytest.raises(ValueError, match="configuration"):
        ContinuousReplayConfig(**{name: value})
