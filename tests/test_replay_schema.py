import random

import pytest

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.schema import ReplayRecord, records_from_game
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


def test_replay_record_rejects_invalid_targets() -> None:
    _, game = scripted_game()
    record = records_from_game(game, run_id="pilot")[0]
    data = record.to_dict()
    data["outcome_value"] = 2
    with pytest.raises(ValueError, match="outcome"):
        ReplayRecord.from_dict(data)

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
