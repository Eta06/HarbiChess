from pathlib import Path

import pytest

from harbichess.core.state import Side
from harbichess.evaluation.deterministic_value_probe import (
    DeterministicValueProbeConfig,
    _pearson,
    _round_robin,
)
from harbichess.replay.schema import ReplayRecord


def _record(game: str, ply: int) -> ReplayRecord:
    return ReplayRecord(
        game_id=game,
        game_index=0,
        seed=1,
        ply=ply,
        root_fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        moves=tuple("a1a2" for _ in range(ply)),
        side_to_move=Side.WHITE if ply % 2 == 0 else Side.BLACK,
        policy=((0, 1.0),),
        selected_action=0,
        root_value=0.0,
        outcome_value=0,
        repetition_redirected=False,
    )


def test_round_robin_covers_games_before_deeper_positions() -> None:
    records = tuple(
        _record(game, ply) for game in ("a", "b", "c") for ply in range(4)
    )

    selected = _round_robin(records, 5)

    assert [(record.game_id, record.ply) for record in selected] == [
        ("a", 0),
        ("b", 0),
        ("c", 0),
        ("a", 1),
        ("b", 1),
    ]


def test_pearson_detects_deterministic_order() -> None:
    assert _pearson([-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]) == pytest.approx(1.0)
    assert _pearson([-1.0, 0.0, 1.0], [0.5, 0.0, -0.5]) == pytest.approx(-1.0)


def test_probe_config_rejects_unaligned_schedule() -> None:
    with pytest.raises(ValueError, match="configuration"):
        DeterministicValueProbeConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            steps=21,
            validation_interval=20,
        )
