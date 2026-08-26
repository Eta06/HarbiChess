from types import SimpleNamespace

from harbichess.evaluation.value_pipeline_diagnostics import legacy_max_ply_game_ids


def test_legacy_max_ply_detection_is_schema_gated() -> None:
    records = tuple(
        SimpleNamespace(game_id="truncated", ply=ply, outcome_value=0) for ply in range(4)
    )
    decisive = tuple(
        SimpleNamespace(game_id="decisive", ply=ply, outcome_value=1) for ply in range(4)
    )

    assert legacy_max_ply_game_ids(
        (*records, *decisive),
        max_plies=4,
        target_schema=9,
    ) == frozenset({"truncated"})
    assert not legacy_max_ply_game_ids(
        records,
        max_plies=4,
        target_schema=10,
    )
