import pytest

from harbichess.evaluation.quality import estimate_arena_quality


def test_balanced_arena_has_zero_elo_gain() -> None:
    quality = estimate_arena_quality(50, 100, 50, minimum_games=200)

    assert quality.games == 200
    assert quality.score_rate == 0.5
    assert quality.elo_delta == pytest.approx(0.0)
    assert quality.elo_low < 0 < quality.elo_high
    assert not quality.promotion_ready


def test_clear_arena_win_can_promote_candidate() -> None:
    quality = estimate_arena_quality(140, 40, 20, minimum_games=200)

    assert quality.score_rate == 0.8
    assert quality.elo_delta == pytest.approx(240.824, abs=0.001)
    assert quality.elo_low > 0
    assert quality.promotion_ready


def test_empty_arena_and_invalid_counts() -> None:
    assert estimate_arena_quality(0, 0, 0).elo_low is None
    assert estimate_arena_quality(1, 0, 0).elo_low is None
    with pytest.raises(ValueError, match="non-negative"):
        estimate_arena_quality(-1, 0, 0)
