import chess
import pytest

from harbichess.chess.actions import (
    POLICY_SIZE,
    action_destination_square,
    action_to_legal_move,
    legal_action_indices,
    legal_action_mask,
    move_to_action,
)


def test_initial_legal_actions_are_unique_and_masked() -> None:
    board = chess.Board()
    indices = legal_action_indices(board)
    mask = legal_action_mask(board)

    assert len(indices) == 20
    assert len(mask) == POLICY_SIZE == 4_672
    assert sum(mask) == 20


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
    ],
)
def test_every_legal_move_round_trips(fen: str) -> None:
    board = chess.Board(fen)
    for move in board.legal_moves:
        action = move_to_action(board, move)
        assert action_to_legal_move(board, action) == move
        expected_destination = (
            move.to_square if board.turn is chess.WHITE else chess.square_mirror(move.to_square)
        )
        assert action_destination_square(action) == expected_destination


def test_underpromotion_pieces_have_distinct_actions() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    actions = {move_to_action(board, chess.Move.from_uci(f"a7a8{piece}")) for piece in "nbr"}
    assert len(actions) == 3


def test_action_respects_black_side_canonicalization() -> None:
    white = chess.Board()
    black = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert move_to_action(white, chess.Move.from_uci("e2e4")) == move_to_action(
        black, chess.Move.from_uci("e7e5")
    )


def test_non_chess_geometry_is_rejected() -> None:
    board = chess.Board.empty()
    board.turn = chess.WHITE
    with pytest.raises(ValueError, match="invalid ray move geometry"):
        move_to_action(board, chess.Move.from_uci("a1c4"))


def test_action_destination_reports_off_board_geometry() -> None:
    assert action_destination_square(63 * 73) is None
    with pytest.raises(ValueError, match="out of range"):
        action_destination_square(POLICY_SIZE)
