from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.replay.schema import records_from_game
from harbichess.training.batch import GameBalancedSampler, build_training_batch


def test_training_batch_encodes_policy_and_side_to_move_wdl() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")

    batch = build_training_batch(records)

    assert len(batch.positions) == 4
    assert batch.positions[0].shape == (8, 8, 104)
    assert batch.wdl_targets == (2, 0, 2, 0)
    assert all(sum(policy) == pytest.approx(1.0) for policy in batch.policy_targets)


def test_game_balanced_sampler_is_deterministic_and_restorable() -> None:
    _, game = scripted_game()
    first_game = records_from_game(game, run_id="pilot")
    second_game = tuple(
        replace(record, game_id="pilot-000000000008", game_index=8)
        for record in first_game
    )
    sampler = GameBalancedSampler((*first_game, *second_game), seed=7)
    state = sampler.rng_state
    first = sampler.sample(2)
    sampler.set_rng_state(state)
    second = sampler.sample(2)

    assert first == second
    assert len({record.game_id for record in first}) == 2
    with pytest.raises(ValueError, match="batch_size"):
        sampler.sample(0)
