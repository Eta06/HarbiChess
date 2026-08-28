import math
import random
from dataclasses import replace

import pytest

from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.schema import (
    BranchValueEstimate,
    ContinuationEvidence,
    PolicyRegretAdjustment,
    RepetitionRiskEstimate,
    ReplayRecord,
    records_from_game,
)
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import play_game


class ScriptedSearch:
    moves = (ChessMove("f2f3"), ChessMove("e7e5"), ChessMove("g2g4"), ChessMove("d8h4"))

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del rng, add_root_noise
        move = self.moves[state.ply]
        return SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)


def scripted_game():
    rules = PythonChessRules()
    return rules, play_game(
        ScriptedSearch(),
        rules,
        rules.initial_state(),
        game_index=7,
        seed=99,
    )


def test_self_play_game_converts_to_legal_versioned_records() -> None:
    rules, game = scripted_game()
    records = records_from_game(game, run_id="pilot", rules=rules)

    assert len(records) == 4
    assert records[0].game_id == "pilot-000000000007"
    assert records[0].side_to_move.value == "white"
    assert records[0].outcome_value == -1
    assert records[0].selected_action == records[0].policy[0][0]
    assert ReplayRecord.from_dict(records[2].to_dict()) == records[2]


def test_replay_record_round_trips_teacher_policy_evidence() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    evidenced = replace(
        record,
        raw_policy=record.policy,
        teacher_policy_tv=0.0,
        teacher_policy_kl=0.0,
        teacher_argmax_changed=False,
        teacher_search_value_delta=0.0,
    )

    assert ReplayRecord.from_dict(evidenced.to_dict()) == evidenced
    evidenced.validate_rules(rules)

    invalid = evidenced.to_dict()
    invalid["teacher_policy_tv"] = 1.1
    with pytest.raises(ValueError, match="outside"):
        ReplayRecord.from_dict(invalid)


def test_replay_allows_explicitly_decoupled_behavior_action() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    board = rules.board(record.state)
    alternative = move_to_action(board, board.parse_uci("e2e4"))
    decoupled = replace(
        record,
        policy=((alternative, 1.0),),
        behavior_target_decoupled=True,
    )

    decoupled.validate_rules(rules)
    assert ReplayRecord.from_dict(decoupled.to_dict()) == decoupled


def test_replay_record_rejects_invalid_targets() -> None:
    _, game = scripted_game()
    record = records_from_game(game, run_id="pilot")[0]
    data = record.to_dict()
    data["outcome_value"] = 2
    with pytest.raises(ValueError, match="outcome"):
        ReplayRecord.from_dict(data)

    data = record.to_dict()
    data["outcome_value"] = None
    assert ReplayRecord.from_dict(data).outcome_value is None

    data = record.to_dict()
    data["policy"] = ((record.selected_action, 0.5),)
    with pytest.raises(ValueError, match="normalized"):
        ReplayRecord.from_dict(data)

    data = record.to_dict()
    data["policy"] = (
        (record.selected_action, 0.0),
        ((record.selected_action + 1) % POLICY_SIZE, 1.0),
    )
    with pytest.raises(ValueError, match="positive"):
        ReplayRecord.from_dict(data)


def test_replay_record_round_trips_root_search_confidence() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    adjusted = replace(
        record,
        root_search_adjusted=True,
        root_search_first_margin=0.12,
        root_search_final_margin=0.18,
    )

    assert ReplayRecord.from_dict(adjusted.to_dict()) == adjusted

    legacy = adjusted.to_dict()
    for field in (
        "root_search_adjusted",
        "root_search_first_margin",
        "root_search_final_margin",
    ):
        legacy.pop(field)
    assert not ReplayRecord.from_dict(legacy).root_search_adjusted

    invalid = adjusted.to_dict()
    invalid["root_search_final_margin"] = None
    with pytest.raises(ValueError, match="both confidence margins"):
        ReplayRecord.from_dict(invalid)


def test_replay_record_round_trips_branch_confidence_evidence() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    board = rules.board(record.state)
    repeat_action = move_to_action(board, board.parse_uci("e2e4"))
    evidence = ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        minimum_advantage=0.01,
        repeat_actions=(repeat_action,),
        branches=(
            BranchValueEstimate(
                action=record.selected_action,
                move="f2f3",
                samples=8,
                mean_value=0.20,
                standard_error=0.04,
                lower_confidence_bound=0.12,
                upper_confidence_bound=0.28,
            ),
        ),
        qualified_actions=(record.selected_action,),
        source_model_sha256="a" * 64,
    )
    evidenced = replace(record, continuation_evidence=evidence)

    assert ReplayRecord.from_dict(evidenced.to_dict()) == evidenced

    payload = evidenced.to_dict()
    payload["continuation_evidence"]["qualified_actions"] = [repeat_action]
    with pytest.raises(ValueError, match="qualified"):
        ReplayRecord.from_dict(payload)


def test_replay_record_round_trips_multi_ply_repetition_risk() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    board = rules.board(record.state)
    repeat_action = move_to_action(board, board.parse_uci("e2e4"))
    branch = BranchValueEstimate(
        action=record.selected_action,
        move="f2f3",
        samples=8,
        mean_value=0.20,
        standard_error=0.04,
        lower_confidence_bound=0.12,
        upper_confidence_bound=0.28,
    )
    risk = RepetitionRiskEstimate(
        action=record.selected_action,
        horizon_plies=3,
        rollouts=16,
        repetition_events=0,
        estimated_risk=0.0,
        upper_confidence_bound=0.145,
    )
    evidence = ContinuationEvidence(
        method_version=2,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        minimum_advantage=0.01,
        repeat_actions=(repeat_action,),
        branches=(branch,),
        qualified_actions=(record.selected_action,),
        source_model_sha256="a" * 64,
        repetition_risks=(risk,),
        maximum_repetition_risk=0.25,
    )
    evidenced = replace(record, continuation_evidence=evidence)

    assert ReplayRecord.from_dict(evidenced.to_dict()) == evidenced

    legacy_v5 = evidenced.to_dict()
    legacy_risk = legacy_v5["continuation_evidence"]["repetition_risks"][0]
    for field in (
        "loop_value_samples",
        "exact_loop_value_samples",
        "mean_loop_value",
        "lower_loop_value_bound",
        "risk_adjusted_value_lower_bound",
    ):
        legacy_risk.pop(field)
    legacy_v5["continuation_evidence"].pop("evaluated_root_value")
    legacy_v5["continuation_evidence"].pop("minimum_advantaged_root_value")
    assert ReplayRecord.from_dict(legacy_v5) == evidenced

    payload = evidenced.to_dict()
    payload["continuation_evidence"]["maximum_repetition_risk"] = 0.10
    with pytest.raises(ValueError, match="risk gate"):
        ReplayRecord.from_dict(payload)


def test_replay_record_round_trips_continuous_policy_regret() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="pilot", rules=rules)[0]
    board = rules.board(record.state)
    repeat_action = move_to_action(board, board.parse_uci("e2e4"))
    regret = 0.04
    adjustment = PolicyRegretAdjustment(
        method_version=1,
        temperature=0.02,
        root_value=0.04,
        repeat_value=0.0,
        best_nonrepeat_value=0.14,
        regret=regret,
        redirect_fraction=1.0 - math.exp(-regret / 0.02),
        repeat_actions=(repeat_action,),
        redirect_actions=(record.selected_action,),
        source_model_sha256="b" * 64,
    )
    adjusted = replace(
        record,
        policy=((repeat_action, 0.25), (record.selected_action, 0.75)),
        root_value=0.04,
        repetition_redirected=True,
        policy_regret_adjustment=adjustment,
    )

    assert ReplayRecord.from_dict(adjusted.to_dict()) == adjusted
    adjusted.validate_rules(rules)
