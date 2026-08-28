from dataclasses import replace

from test_replay_schema import scripted_game

from harbichess.chess.rules import PythonChessRules
from harbichess.replay.coverage import ReplayCoverageThresholds, measure_replay_coverage
from harbichess.replay.schema import records_from_game


def test_replay_coverage_qualifies_teacher_telemetry_with_frozen_thresholds() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="coverage", rules=rules)[0]
    evidenced = replace(
        record,
        raw_policy=record.policy,
        teacher_policy_tv=0.0,
        teacher_policy_kl=0.0,
        teacher_argmax_changed=False,
        teacher_search_value_delta=0.1,
    )
    thresholds = ReplayCoverageThresholds(
        minimum_samples=1,
        minimum_unique_position_ratio=1.0,
        minimum_opening_ratio=1.0,
        minimum_middlegame_ratio=0.0,
        minimum_endgame_ratio=0.0,
        minimum_tactical_ratio=0.0,
        minimum_quiet_ratio=0.0,
        minimum_value_bucket_ratio=0.0,
        minimum_outcome_bucket_ratio=0.0,
        minimum_material_signatures=1,
        minimum_position_signatures=1,
        minimum_teacher_telemetry_ratio=1.0,
        minimum_comparable_teacher_deltas=1,
        minimum_positive_teacher_delta_ratio=1.0,
    )

    report = measure_replay_coverage((evidenced,), thresholds=thresholds, rules=rules)

    assert report.passed
    assert report.teacher_telemetry_samples == 1
    assert report.positive_teacher_delta_ratio == 1.0


def test_replay_coverage_rejects_missing_teacher_evidence() -> None:
    rules = PythonChessRules()
    _, game = scripted_game()
    record = records_from_game(game, run_id="coverage", rules=rules)[0]
    thresholds = ReplayCoverageThresholds(
        minimum_samples=1,
        minimum_unique_position_ratio=0.0,
        minimum_opening_ratio=0.0,
        minimum_middlegame_ratio=0.0,
        minimum_endgame_ratio=0.0,
        minimum_tactical_ratio=0.0,
        minimum_quiet_ratio=0.0,
        minimum_value_bucket_ratio=0.0,
        minimum_outcome_bucket_ratio=0.0,
        minimum_material_signatures=1,
        minimum_position_signatures=1,
        minimum_teacher_telemetry_ratio=1.0,
        minimum_comparable_teacher_deltas=1,
        minimum_positive_teacher_delta_ratio=0.0,
        minimum_mean_teacher_delta=-1.0,
    )

    report = measure_replay_coverage((record,), thresholds=thresholds, rules=rules)

    assert not report.passed
    assert "teacher telemetry coverage below threshold" in report.reasons
