import pytest

from harbichess.core.state import ChessMove, GameOutcome, Side, TerminalResult


def test_move_normalizes_uci() -> None:
    assert ChessMove(" E7E8Q ").uci == "e7e8q"


def test_move_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="invalid UCI move length"):
        ChessMove("e2")


@pytest.mark.parametrize(
    ("result", "side", "expected"),
    [
        (TerminalResult.WHITE_WIN, Side.WHITE, 1),
        (TerminalResult.WHITE_WIN, Side.BLACK, -1),
        (TerminalResult.BLACK_WIN, Side.WHITE, -1),
        (TerminalResult.DRAW, Side.BLACK, 0),
    ],
)
def test_outcome_value_is_perspective_correct(
    result: TerminalResult, side: Side, expected: int
) -> None:
    outcome = GameOutcome(result=result, termination="test")
    assert outcome.value_for(side) == expected

