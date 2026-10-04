import random
from dataclasses import asdict, replace

import chess
import pytest
from rules_v2_original_reference import IllegalMoveError
from rules_v2_original_reference import PythonChessRules as OldRules

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import IllegalMoveError as NewIllegalMoveError
from harbichess.chess.rules import PythonChessRules as NewRules
from harbichess.core.state import ChessMove, ChessState


def history(moves, fen=chess.STARTING_FEN):
    return ChessState(fen, tuple(ChessMove(move) for move in moves))


def random_states(count=128, plies=80):
    states = []
    rng = random.Random(6719)
    for _ in range(count):
        board = chess.Board()
        moves = []
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            board.push(move)
            moves.append(move.uci())
        states.append(history(moves))
    return states


def test_cold_miss_remembers_only_final_requested_state_and_hot_games_survive():
    states = random_states()
    rules = NewRules(board_cache_size=512)
    for state in states:
        rules.inspect(state)
    assert len(rules._cache()) == len(states)
    assert set(rules._cache()) == set(states)
    for state in states:
        cached = rules._cache()[state]
        assert rules.inspect(state) is cached
        assert len(cached.move_stack) == state.ply


def test_nearest_cached_ancestor_is_isolated_and_only_missing_suffix_is_pushed(monkeypatch):
    rules = NewRules(board_cache_size=4)
    parent = history(["e2e4", "e7e5"])
    parent_board = rules.inspect(parent)
    child = history(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
    calls = []
    real_push = chess.Board.push

    def push(board, move):
        calls.append(move.uci())
        return real_push(board, move)

    monkeypatch.setattr(chess.Board, "push", push)
    actual = rules.inspect(child)
    assert calls == ["g1f3", "b8c6", "f1b5"]
    assert len(parent_board.move_stack) == 2 and len(actual.move_stack) == 5
    assert set(rules._cache()) == {parent, child}
    external = rules.board(child)
    external.push(chess.Move.from_uci("a7a6"))
    assert len(rules.inspect(child).move_stack) == 5


@pytest.mark.parametrize(
    "state",
    [
        history([]),
        history(["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]),
        history(["e1g1", "e8c8"], "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        history(["g1f3", "g8f6", "f3g1", "f6g8"] * 2),
        history(["f2f3", "e7e5", "g2g4", "d8h4"]),
        history(["b1b2"], "7k/8/8/8/8/8/8/KR6 w - - 98 1"),
        history(["a7a8q"], "7k/P7/8/8/8/8/8/7K w - - 0 1"),
    ],
)
def test_full_history_board_facts_claims_and_all104_inputs_are_exact(state):
    old, new = OldRules(board_cache_size=2), NewRules(board_cache_size=2)
    left, right = old.board(state), new.board(state)
    assert left.fen(en_passant="fen") == right.fen(en_passant="fen")
    assert left.move_stack == right.move_stack
    assert old.legal_moves(state) == new.legal_moves(state)
    assert asdict(old.view(state)) == asdict(new.view(state))
    for claim_draw in (False, True):
        assert old.outcome(state, claim_draw=claim_draw) == new.outcome(
            state, claim_draw=claim_draw
        )
    assert old.claimable_threefold_moves(state, old.legal_moves(state)) == (
        new.claimable_threefold_moves(state, new.legal_moves(state))
    )
    assert BoardEncoder(old).encode(state) == BoardEncoder(new).encode(state)


@pytest.mark.parametrize(
    "state",
    [
        history(["e2e4", "e7e5", "e2e5"]),
        history(["e2e4", "xxxx"]),
        history(["e2e5"]),
        ChessState("invalid fen"),
    ],
)
def test_invalid_root_and_move_errors_match_and_no_partial_prefix_pollution(state):
    old, new = OldRules(), NewRules()
    errors = []
    for rules in (old, new):
        with pytest.raises((ValueError, IllegalMoveError, NewIllegalMoveError)) as caught:
            rules.inspect(state)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
    assert not new._cache()


def test_128_long_random_game_inputs_match_with_repeated_cache_eviction():
    old, new = OldRules(board_cache_size=512), NewRules(board_cache_size=512)
    old_encoder, new_encoder = BoardEncoder(old, cache_size=1), BoardEncoder(new, cache_size=1)
    for state in random_states():
        assert old_encoder.encode(state) == new_encoder.encode(state)
        assert old.board(state).move_stack == new.board(state).move_stack
    assert len(new._cache()) == 128


def test_cold_replay_above_python_recursion_limit_keeps_full_stack():
    # Legal repetitive moves, intentionally replayable beyond terminal claim.
    state = history(["g1f3", "g8f6", "f3g1", "f6g8"] * 300)
    rules = NewRules(board_cache_size=1)
    board = rules.board(state)
    assert len(board.move_stack) == 1200
    assert board.is_fivefold_repetition()
    assert len(rules._cache()) == 1


def test_actual_next_updates_old_and_new_rules_have_identical_trajectories(tmp_path):
    from test_torch_online_checkpoint import equal_tree
    from test_torch_online_learner import SOURCE, config, inputs, make_inputs

    from harbichess.training.torch_online_learner import TorchOnlineLearner

    make_inputs(tmp_path)
    cfg = replace(config(), actors=replace(config().actors, max_additional_plies=24))
    first = TorchOnlineLearner.fresh(config=cfg, input_paths=inputs(tmp_path), source_commit=SOURCE)
    second = TorchOnlineLearner.fresh(
        config=cfg, input_paths=inputs(tmp_path), source_commit=SOURCE
    )
    first.actors.rules = OldRules(board_cache_size=8)
    second.actors.rules = NewRules(board_cache_size=8)
    first.encoder, second.encoder = (
        BoardEncoder(first.actors.rules),
        BoardEncoder(second.actors.rules),
    )
    for _ in range(6):
        assert first.train_update() == second.train_update()
        assert first.actors.cursor() == second.actors.cursor()
        assert first.actors.rng.getstate() == second.actors.rng.getstate()
        for left, right in (
            (first.online, second.online),
            (first.base, second.base),
            (first.ema, second.ema),
        ):
            equal_tree(left.state_dict(), right.state_dict())
        equal_tree(first.optimizer.state_dict(), second.optimizer.state_dict())
