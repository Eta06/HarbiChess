import pytest
from test_replay_schema import scripted_game

from harbichess.evaluation.teacher_qualification import (
    QualificationConfig,
    _mean_policy,
    _tv,
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
