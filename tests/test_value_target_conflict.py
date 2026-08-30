from types import SimpleNamespace

from harbichess.evaluation.value_target_conflict import _distribution, _summarize_keyed


def _row(game: str, state: str, outcome: int, ply: int = 0):
    return SimpleNamespace(game_id=game, state_key=state, outcome_value=outcome, ply=ply)


def test_state_audit_detects_cross_domain_conflicting_targets() -> None:
    historical = (_row("old-a", "same", 1), _row("old-b", "old-only", 0))
    fresh = (_row("new-a", "same", -1), _row("new-b", "new-only", 0))

    result = _summarize_keyed(historical, fresh, lambda row: row.state_key)

    assert result["unique_states"] == 3
    assert result["overlapping_states"] == 1
    assert result["cross_domain_conflicted_states"] == 1
    assert result["mean_conflict_entropy_bits"] == 1.0


def test_distribution_reports_rows_games_outcomes_and_phases() -> None:
    records = (
        _row("a", "x", 1, 3),
        _row("a", "y", -1, 25),
        _row("b", "z", 0, 90),
    )

    result = _distribution(records)

    assert result == {
        "rows": 3,
        "games": 2,
        "outcomes": {-1: 1, 0: 1, 1: 1},
        "phases": {"endgame": 1, "middlegame": 1, "opening": 1},
    }
