import json

import pytest
from test_replay_schema import scripted_game

from harbichess.dashboard.state import SnapshotStore
from harbichess.evaluation.teacher_qualification import (
    QualificationConfig,
    _mean_policy,
    _tv,
    publish_qualification_result,
    select_stratified_records,
)
from harbichess.replay.schema import records_from_game


def test_stratified_position_selection_is_deterministic_and_bounded(tmp_path) -> None:
    rules, game = scripted_game()
    records = records_from_game(game, run_id="qualification", rules=rules)

    first = select_stratified_records(records, rules=rules, count=3, seed=17)
    second = select_stratified_records(records, rules=rules, count=3, seed=17)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    with pytest.raises(ValueError, match="positive"):
        select_stratified_records(records, rules=rules, count=0, seed=17)


def test_policy_averaging_preserves_probability_and_measures_seed_drift() -> None:
    rules, game = scripted_game()
    first, second = records_from_game(game, run_id="qualification", rules=rules)[:2]
    first_move = rules.legal_moves(first.state)[0]
    second_move = rules.legal_moves(second.state)[0]
    policies = (
        ((first_move, 1.0),),
        ((first_move, 0.5), (second_move, 0.5)),
    )

    averaged = _mean_policy(policies)

    assert sum(probability for _, probability in averaged) == pytest.approx(1.0)
    assert _tv(policies[0], policies[1]) == pytest.approx(0.5)


def test_qualification_config_rejects_empty_replay(tmp_path) -> None:
    with pytest.raises(ValueError, match="configuration"):
        QualificationConfig(
            run_result=tmp_path / "result.json",
            shards=(),
            output_dir=tmp_path / "qualification",
        )


def test_completed_qualification_blocks_dashboard_learner_on_failure(tmp_path) -> None:
    result_path = tmp_path / "qualification" / "qualification.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "gate": {"qualified": False, "qualified_variants": []},
                "selection": {"selected_records": 32},
                "raw_value_mse": 0.5,
                "variants": {
                    "raw": {"mean_verified_action_value_delta": 0.0},
                    "puct-64-clean": {
                        "mean_verified_action_value_delta": -0.01,
                        "verified_action_value_delta_95_interval": [-0.02, 0.0],
                        "mean_seed_stability_tv": 0.0,
                        "value_mse": 0.49,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "state.json"

    publish_qualification_result(result_path, telemetry)

    snapshot = SnapshotStore(telemetry).read()
    assert snapshot.teacher_qualification_status == "failed"
    assert snapshot.teacher_best_variant == "puct-64-clean"
    assert "blocked" in snapshot.mode_detail
    assert not snapshot.promotion_ready
