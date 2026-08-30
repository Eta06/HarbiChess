from pathlib import Path

import pytest

from harbichess.core.state import Side
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _split_games,
    _trajectory_fingerprint,
    _white_outcome,
)
from harbichess.replay.schema import ReplayRecord


def _game(
    game_id: str, winner: int, length: int = 4, *, trajectory_id: str | None = None
) -> tuple[ReplayRecord, ...]:
    moves = ("a2a3", "a7a6", "b2b3", "b7b6")
    return tuple(
        ReplayRecord(
            game_id=game_id,
            game_index=0,
            seed=1,
            ply=ply,
            root_fen=f"trajectory-{trajectory_id or game_id}",
            moves=moves[:ply],
            side_to_move=Side.WHITE if ply % 2 == 0 else Side.BLACK,
            policy=((ply, 1.0),),
            selected_action=ply,
            root_value=0.0,
            outcome_value=(
                0
                if winner == 0
                else winner * (1 if ply % 2 == 0 else -1)
            ),
            repetition_redirected=False,
        )
        for ply in range(length)
    )


def test_trajectory_fingerprint_ignores_run_local_game_identity() -> None:
    first = _game("run-a-game-1", 1, trajectory_id="shared")
    second = tuple(
        ReplayRecord(
            **{
                **record.to_dict(),
                "game_id": "run-b-game-9",
                "side_to_move": record.side_to_move,
            }
        )
        for record in first
    )

    assert _trajectory_fingerprint(first) == _trajectory_fingerprint(second)


def test_white_outcome_checks_side_to_move_perspective() -> None:
    assert _white_outcome(_game("white", 1)) == 1
    assert _white_outcome(_game("black", -1)) == -1
    assert _white_outcome(_game("draw", 0)) == 0


def test_split_is_deterministic_stratified_and_trajectory_disjoint() -> None:
    games = {
        _trajectory_fingerprint(game): game
        for outcome in (-1, 0, 1)
        for index in range(4)
        for game in (_game(f"{outcome}-{index}", outcome, length=index + 1),)
    }

    first = _split_games(games, seed=17)
    second = _split_games(games, seed=17)

    assert first == second
    train, validation, audit = first
    assert audit["train_trajectories"] == 9
    assert audit["validation_trajectories"] == 3
    assert audit["fingerprint_overlap"] == 0
    assert set(audit["validation_outcomes_white"]) == {"-1", "0", "1"}
    assert not ({record.game_id for record in train} & {record.game_id for record in validation})


def test_config_rejects_duplicate_sources_and_unaligned_schedule() -> None:
    with pytest.raises(ValueError, match="unique"):
        CorrectedReplayValueTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            run_ids=("same", "same"),
        )
    with pytest.raises(ValueError, match="configuration"):
        CorrectedReplayValueTransferConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            steps=21,
            validation_interval=20,
        )
