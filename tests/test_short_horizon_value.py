from __future__ import annotations

from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.replay.schema import records_from_game
from harbichess.training.short_horizon_value import short_horizon_targets


def test_short_horizon_targets_flip_perspective_between_plies() -> None:
    rules, game = scripted_game()
    source = records_from_game(game, run_id="short", rules=rules)[:2]
    records = (
        replace(source[0], root_value=0.2, outcome_value=1),
        replace(source[1], root_value=0.4, outcome_value=-1),
    )

    targets = short_horizon_targets(records, coefficient=0.8)

    final_target = 0.2 * 0.4 + 0.8 * -1.0
    assert targets[1] == pytest.approx(final_target)
    assert targets[0] == pytest.approx(0.2 * 0.2 - 0.8 * final_target)


def test_short_horizon_max_ply_tail_uses_last_root_value() -> None:
    rules, game = scripted_game()
    source = records_from_game(game, run_id="short", rules=rules)[:1]
    record = replace(source[0], root_value=0.5, outcome_value=None)

    assert short_horizon_targets((record,), coefficient=0.8) == pytest.approx((0.5,))


def test_short_horizon_rejects_nonconsecutive_game_rows() -> None:
    rules, game = scripted_game()
    source = records_from_game(game, run_id="short", rules=rules)

    with pytest.raises(ValueError, match="consecutive"):
        short_horizon_targets((source[0], source[2]), coefficient=0.8)
