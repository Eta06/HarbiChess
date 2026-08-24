from dataclasses import replace
from pathlib import Path

from harbichess.dashboard.state import RunMode, SnapshotStore, demo_snapshot, empty_snapshot


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

