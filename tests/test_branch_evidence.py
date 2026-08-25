from test_replay_schema import scripted_game

from harbichess.chess.actions import move_to_action
from harbichess.replay.branch_evidence import build_confidence_target, confidence_estimate
from harbichess.replay.schema import BranchValueEstimate, ContinuationEvidence, records_from_game


def test_confidence_estimate_uses_repeated_branch_values() -> None:
    estimate = confidence_estimate(
        action=7,
        move="a2a3",
        values=(0.10, 0.20, 0.30, 0.20),
        confidence_level=0.95,
        comparisons=3,
    )

    assert estimate.mean_value == 0.20
    assert estimate.standard_error > 0
    assert estimate.lower_confidence_bound < estimate.mean_value
    assert estimate.upper_confidence_bound > estimate.mean_value


def test_confidence_target_uses_only_positive_lower_bounds() -> None:
    rules, game = scripted_game()
    record = records_from_game(game, run_id="branch", rules=rules)[0]
    board = rules.board(record.state)
    first = record.selected_action
    second = move_to_action(board, board.parse_uci("e2e4"))
    repeat = move_to_action(board, board.parse_uci("d2d4"))
    branches = (
        BranchValueEstimate(first, "f2f3", 8, 0.25, 0.02, 0.20, 0.30),
        BranchValueEstimate(second, "e2e4", 8, 0.15, 0.02, 0.10, 0.20),
    )
    evidence = ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        repeat_actions=(repeat,),
        branches=branches,
        qualified_actions=(first, second),
        source_model_sha256="a" * 64,
    )

    target = build_confidence_target(record, evidence)

    assert target is not None
    assert target.selected_action == first
    assert dict(target.policy)[first] == 2 / 3
    assert dict(target.policy)[second] == 1 / 3
    assert target.continuation_evidence == evidence

    rejected = ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        repeat_actions=(repeat,),
        branches=branches,
        qualified_actions=(),
        source_model_sha256="a" * 64,
    )
    assert build_confidence_target(record, rejected) is None
