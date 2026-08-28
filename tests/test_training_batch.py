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
    assert batch.value_weights == (1.0, 1.0, 1.0, 1.0)
    assert all(sum(policy) == pytest.approx(1.0) for policy in batch.policy_targets)
    assert all(any(mask) for mask in batch.legal_masks)
    assert all(
        not probability or mask
        for policy, legal in zip(batch.policy_targets, batch.legal_masks, strict=True)
        for probability, mask in zip(policy, legal, strict=True)
    )


def test_game_balanced_sampler_is_deterministic_and_restorable() -> None:
    _, game = scripted_game()
    first_game = records_from_game(game, run_id="pilot")
    second_game = tuple(
        replace(record, game_id="pilot-000000000008", game_index=8) for record in first_game
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


def test_prevalidated_batch_selection_reuses_encoded_rows() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    batch = build_training_batch(records)

    selected = batch.select((2, 0, 2))

    assert selected.positions == (batch.positions[2], batch.positions[0], batch.positions[2])
    assert selected.policy_targets == (
        batch.policy_targets[2],
        batch.policy_targets[0],
        batch.policy_targets[2],
    )
    assert selected.legal_masks == (
        batch.legal_masks[2],
        batch.legal_masks[0],
        batch.legal_masks[2],
    )
    assert selected.wdl_targets == (
        batch.wdl_targets[2],
        batch.wdl_targets[0],
        batch.wdl_targets[2],
    )
    assert selected.value_weights == (
        batch.value_weights[2],
        batch.value_weights[0],
        batch.value_weights[2],
    )
    with pytest.raises(IndexError, match="indices"):
        batch.select(())


def test_sampler_indices_address_the_original_replay_tuple() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    by_record = GameBalancedSampler(records, seed=19)
    by_index = GameBalancedSampler(records, seed=19)

    sampled = by_record.sample(6)
    indices = by_index.sample_indices(6)

    assert sampled == tuple(records[index] for index in indices)


def test_training_batch_masks_unknown_value_targets_but_keeps_policy() -> None:
    _, game = scripted_game()
    record = replace(records_from_game(game, run_id="truncated")[0], outcome_value=None)

    batch = build_training_batch((record,))

    assert batch.wdl_targets == (1,)
    assert batch.value_weights == (0.0,)
    assert sum(batch.policy_targets[0]) == pytest.approx(1.0)


def test_sampler_holds_requested_continuation_fraction() -> None:
    _, game = scripted_game()
    standard = records_from_game(game, run_id="standard")
    continuation = tuple(
        replace(
            record,
            game_id=f"continuation-{index}",
            game_index=100 + index,
            repetition_redirected=True,
        )
        for index, record in enumerate(standard)
    )
    sampler = GameBalancedSampler(
        (*standard, *continuation),
        seed=17,
        continuation_fraction=0.25,
    )

    sampled = sampler.sample(8)

    assert sum(record.repetition_redirected for record in sampled) == 2


def test_sampler_applies_continuation_recency_weights() -> None:
    _, game = scripted_game()
    standard = records_from_game(game, run_id="standard")
    continuation = tuple(
        replace(
            record,
            game_id=f"continuation-{index}",
            game_index=100 + index,
            repetition_redirected=True,
        )
        for index, record in enumerate(standard)
    )
    weights = {record.game_id: 1.0 for record in continuation}
    weights["continuation-0"] = 100.0
    sampler = GameBalancedSampler(
        (*standard, *continuation),
        seed=23,
        continuation_fraction=0.5,
        continuation_game_weights=weights,
    )

    sampled = tuple(record for _ in range(100) for record in sampler.sample(4))
    continuation_counts = {
        game_id: sum(record.game_id == game_id for record in sampled) for game_id in weights
    }

    assert continuation_counts["continuation-0"] > sum(
        continuation_counts[game_id]
        for game_id in continuation_counts
        if game_id != "continuation-0"
    )
