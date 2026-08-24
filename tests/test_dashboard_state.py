from dataclasses import replace
from pathlib import Path

from harbichess.dashboard.state import (
    MAX_HISTORY_POINTS,
    CheckpointStatus,
    HistoryPoint,
    PilotStatus,
    RunMode,
    SnapshotStore,
    demo_snapshot,
    empty_snapshot,
)


def test_snapshot_round_trip_preserves_nested_game(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = SnapshotStore(path)
    expected = demo_snapshot()
    store.write_atomic(expected)

    assert store.read() == expected
    assert list(tmp_path.iterdir()) == [path]


def test_missing_snapshot_returns_idle_state(tmp_path: Path) -> None:
    snapshot = SnapshotStore(tmp_path / "missing.json").read()
    assert snapshot.mode is RunMode.IDLE
    assert snapshot.lifetime_games == 0


def test_training_elapsed_time_is_persisted() -> None:
    snapshot = replace(empty_snapshot(), training_elapsed_seconds=7_200.0)
    assert snapshot.from_json(snapshot.to_json()).training_elapsed_seconds == 7_200.0


def test_history_is_bounded_and_round_trips() -> None:
    point = HistoryPoint(1, 2.0, 3, 4.0, 5.0, 1.0, 9.0, 6.0, 7.0)
    snapshot = empty_snapshot()
    for _ in range(MAX_HISTORY_POINTS + 5):
        snapshot = snapshot.append_history(point)

    restored = snapshot.from_json(snapshot.to_json())
    assert len(restored.history) == MAX_HISTORY_POINTS
    assert restored.history[-1] == point


def test_demo_snapshot_contains_arena_quality() -> None:
    snapshot = demo_snapshot()
    assert snapshot.arena_games == 220
    assert snapshot.arena_elo_low > 0
    assert snapshot.promotion_ready
    assert snapshot.pilot_status is PilotStatus.TRAINING
    assert snapshot.pilot_stopped_early
    assert snapshot.pilot_best_validation_step == 18_400
    assert snapshot.checkpoint_status is CheckpointStatus.VERIFIED
    assert snapshot.diversity.openings[-1].ply == 12
    assert snapshot.diversity.terminations[0].termination == "checkmate"
    assert snapshot.arena_threefold_repetitions == 41
    assert snapshot.arena_avoidable_threefold_repetitions == 38
