import chess
import pytest

from harbichess.chess.rules import IllegalMoveError, PythonChessRules
from harbichess.core.state import ChessMove, Side, TerminalResult


@pytest.fixture
def rules() -> PythonChessRules:
    return PythonChessRules()


def play(rules: PythonChessRules, moves: list[str]):
    state = rules.initial_state()
    for move in moves:
        state = rules.apply(state, ChessMove(move))
    return state


def perft(rules: PythonChessRules, state, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        perft(rules, rules.apply(state, move), depth - 1)
        for move in rules.legal_moves(state)
    )


def test_initial_position_has_twenty_legal_moves(rules: PythonChessRules) -> None:
    assert len(rules.legal_moves(rules.initial_state())) == 20


@pytest.mark.parametrize(("depth", "nodes"), [(1, 20), (2, 400), (3, 8_902)])
def test_initial_position_perft(rules: PythonChessRules, depth: int, nodes: int) -> None:
    assert perft(rules, rules.initial_state(), depth) == nodes


def test_fools_mate_is_checkmate(rules: PythonChessRules) -> None:
    state = play(rules, ["f2f3", "e7e5", "g2g4", "d8h4"])
    outcome = rules.outcome(state)

    assert outcome is not None
    assert outcome.result is TerminalResult.BLACK_WIN
    assert outcome.termination == "checkmate"
    assert rules.legal_moves(state) == ()
    assert rules.view(state).in_check


def test_stalemate_is_not_check(rules: PythonChessRules) -> None:
    state = rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    outcome = rules.outcome(state)

    assert outcome is not None
    assert outcome.result is TerminalResult.DRAW
    assert outcome.termination == "stalemate"
    assert not rules.view(state).in_check


def test_castling_and_en_passant_are_legal(rules: PythonChessRules) -> None:
    castling = rules.initial_state("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert ChessMove("e1g1") in rules.legal_moves(castling)
    assert ChessMove("e1c1") in rules.legal_moves(castling)

    en_passant = play(rules, ["e2e4", "a7a6", "e4e5", "d7d5"])
    assert ChessMove("e5d6") in rules.legal_moves(en_passant)


def test_illegal_move_is_rejected(rules: PythonChessRules) -> None:
    with pytest.raises(IllegalMoveError, match="illegal move"):
        rules.apply(rules.initial_state(), ChessMove("e2e5"))


def test_incremental_board_cache_preserves_external_mutation_isolation(
    rules: PythonChessRules,
) -> None:
    state = play(rules, ["e2e4", "e7e5", "g1f3"])
    expected_fen = rules.board(state).fen()

    leaked = rules.board(state)
    leaked.push(chess.Move.from_uci("b8c6"))

    assert rules.board(state).fen() == expected_fen
    assert rules.view(state).side_to_move is Side.BLACK


def test_board_cache_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cache size"):
        PythonChessRules(board_cache_size=0)


def test_state_replay_preserves_repetition_history(rules: PythonChessRules) -> None:
    state = play(
        rules,
        ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"],
    )

    assert rules.outcome(state, claim_draw=False) is None
    claimed = rules.outcome(state, claim_draw=True)
    assert claimed is not None
    assert claimed.result is TerminalResult.DRAW
    assert claimed.termination == "threefold_repetition"
    assert rules.view(state).side_to_move is Side.WHITE
