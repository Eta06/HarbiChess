import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from own8_audit_core import audit_certified_root_receipt

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessState


def fixture():
    rules = PythonChessRules()
    state = ChessState("6rk/8/8/8/8/6q1/6B1/6K1 w - - 0 1")
    board = rules.inspect(state)
    actions = tuple(legal_action_indices(board))
    moves = {move_to_action(board, m): m.uci() for m in board.legal_moves}
    raw = [0.75 if moves[a] == "g1h1" else 0.25 for a in actions]
    final = [1e-12 if moves[a] == "g1h1" else 1 - 1e-12 for a in actions]
    receipt = dict(
        certified_mates=[],
        moves=[
            dict(move=m, visits=12 if m == "g1h1" else 4, prior=0.5, mean_value=-0.5)
            for m in sorted(moves.values())
        ],
        simulations=16,
        selected_action="g1f1",
        raw_selected_action="g1h1",
        root_value=-0.5,
        search_policy=final,
        raw_search_policy=raw,
        visited_loss_checked_actions=["g1f1", "g1h1"],
        certified_losing_actions=[dict(move="g1h1", opponent_mates=["g3g2"])],
        loss_shield_status="visited-losses-epsilon-shielded",
        loss_shield_epsilon=1e-12,
        selected_action_reason="fallback-visited-nonloss",
    )
    return rules, state, actions, receipt


def check(data):
    rules, state, actions, receipt = data
    return audit_certified_root_receipt(
        state, receipt, actions, rules, claim_draw=True, simulations=16
    )


def test_exact_fullhistory_loss_witness_and_honest_final_policy():
    assert check(fixture()) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("certified_losing_actions", []),
        ("visited_loss_checked_actions", []),
        ("selected_action", "g1h1"),
        ("loss_shield_epsilon", 1e-6),
        ("loss_shield_status", "safe"),
        ("raw_search_policy", [0.6, 0.6]),
    ],
)
def test_corruption_rejected(field, value):
    data = fixture()
    receipt = copy.deepcopy(data[3])
    receipt[field] = value
    with pytest.raises(AssertionError):
        check((*data[:3], receipt))
