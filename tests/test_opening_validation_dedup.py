import json
import random

import chess
import pytest
from harbichess.core.state import ChessMove, ChessState
from harbichess.selfplay.online_actor import ActorOpening, OnlineActorConfig, OnlineActors
from harbichess.training.torch_online_learner import read_online_train_book


def book(tmp_path, rows):
    p = tmp_path / "book.json"
    p.write_text(json.dumps({"schema": 1, "splits": {"train": rows}}))
    return p


def row(source, moves=()):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return {
        "source_game": source,
        "root_ply": len(moves),
        "opening": {"moves": list(moves), "fen": board.fen()},
    }


@pytest.mark.parametrize("field", ["fen", "root_ply", "source_game"])
def test_duplicate_metadata_still_validated(tmp_path, field):
    rows = [row("a", ("e2e4",)), row("b", ("e2e4",))]
    if field == "fen":
        rows[1]["opening"]["fen"] = chess.STARTING_FEN
    elif field == "root_ply":
        rows[1]["root_ply"] = 0
    else:
        rows[1]["source_game"] = "a"
    with pytest.raises(ValueError):
        read_online_train_book(book(tmp_path, rows))


def test_illegal_duplicate_not_cached_as_valid(tmp_path):
    rows = [row("a"), row("b")]
    for entry in rows:
        entry["opening"]["moves"] = ["e2e5"]
        entry["root_ply"] = 1
    with pytest.raises(ValueError):
        read_online_train_book(book(tmp_path, rows))


def test_fullhistory_not_fen_identity_and_claim_repetition(tmp_path):
    repeated = ("g1f3", "g8f6", "f3g1", "f6g8") * 2
    openings = read_online_train_book(book(tmp_path, [row("fresh"), row("repeated", repeated)]))
    assert openings[0].state != openings[1].state
    with pytest.raises(ValueError, match="nonterminal"):
        OnlineActors(openings, config=OnlineActorConfig(2, 16, True, 1.0), rng=random.Random(1))
    actors = OnlineActors(
        openings, config=OnlineActorConfig(2, 16, False, 1.0), rng=random.Random(1)
    )
    assert len(actors.openings) == 2


def test_order_aliases_rng_restore_and_board_isolation(tmp_path):
    moves = ("e2e4", "a7a6", "e4e5", "d7d5")
    openings = read_online_train_book(book(tmp_path, [row("a", moves), row("b", moves), row("c")]))
    rng = random.Random(11)
    actors = OnlineActors(openings, config=OnlineActorConfig(4, 16, True, 1.0), rng=rng)
    assert [o.source_id for o in actors.openings] == ["a", "b", "c"]
    cursor, state = actors.cursor(), rng.getstate()
    restored = OnlineActors(openings, config=actors.config, rng=rng, cursor=cursor)
    assert restored.cursor() == cursor
    assert rng.getstate() == state
    isolated = actors.rules.board(openings[0].state)
    isolated.push_uci("e5d6")
    assert actors.rules.board(openings[0].state).fen() == row("a", moves)["opening"]["fen"]


def test_invalid_root_is_not_hidden_by_other_valid_root():
    invalid = ChessState("8/8/8/8/8/8/8/8 w - - 0 1", ())
    openings = (ActorOpening("valid", ChessState(chess.STARTING_FEN)), ActorOpening("bad", invalid))
    with pytest.raises(ValueError, match="valid"):
        OnlineActors(openings, config=OnlineActorConfig(1, 16, True, 1.0), rng=random.Random(1))


def test_duplicate_terminal_state_rejected():
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    state = ChessState(chess.STARTING_FEN, tuple(ChessMove(m) for m in moves))
    openings = (ActorOpening("a", state), ActorOpening("b", state))
    with pytest.raises(ValueError, match="nonterminal"):
        OnlineActors(openings, config=OnlineActorConfig(1, 16, True, 1.0), rng=random.Random(1))


@pytest.mark.parametrize(
    "moves",
    [
        ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"),
        ("e2e4", "a7a6", "e4e5", "d7d5"),
    ],
)
def test_duplicate_special_state_preserves_exact_legal_history(tmp_path, moves):
    openings = read_online_train_book(book(tmp_path, [row("a", moves), row("b", moves)]))
    actors = OnlineActors(
        openings, config=OnlineActorConfig(1, 16, True, 1.0), rng=random.Random(1)
    )
    expected = chess.Board()
    for move in moves:
        expected.push_uci(move)
    for opening in actors.openings:
        actual = actors.rules.board(opening.state)
        assert actual.fen() == expected.fen()
        assert actual.move_stack == expected.move_stack
        assert set(actual.legal_moves) == set(expected.legal_moves)
        assert actual.can_claim_draw() == expected.can_claim_draw()
        assert opening.state.moves == tuple(ChessMove(m) for m in moves)
