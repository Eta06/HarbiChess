import pytest

from harbichess.chess.encoding import ENCODER_CHANNELS, BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove


def channel_sum(encoded, channel: int) -> float:
    return sum(encoded.values[channel :: ENCODER_CHANNELS])


def test_initial_position_has_expected_shape_and_piece_planes() -> None:
    rules = PythonChessRules()
    encoded = BoardEncoder(rules).encode(rules.initial_state())

    assert encoded.shape == (8, 8, 104)
    assert encoded.schema_version == 1
    assert [channel_sum(encoded, channel) for channel in range(12)] == [
        8.0,
        2.0,
        2.0,
        2.0,
        1.0,
        1.0,
        8.0,
        2.0,
        2.0,
        2.0,
        1.0,
        1.0,
    ]


def test_history_is_encoded_from_current_player_perspective() -> None:
    rules = PythonChessRules()
    state = rules.apply(rules.initial_state(), ChessMove("e2e4"))
    encoded = BoardEncoder(rules).encode(state)

    assert channel_sum(encoded, 0) == 8.0
    assert channel_sum(encoded, 6) == 8.0
    assert channel_sum(encoded, 12) == 8.0
    assert channel_sum(encoded, 18) == 8.0
    assert channel_sum(encoded, 96) == 0.0


def test_repetition_metadata_preserves_game_history() -> None:
    rules = PythonChessRules()
    state = rules.initial_state()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
        state = rules.apply(state, ChessMove(move))
    encoded = BoardEncoder(rules).encode(state)
    assert channel_sum(encoded, 103) == 32.0


def test_prebuilt_board_encoding_matches_state_encoding() -> None:
    rules = PythonChessRules()
    state = rules.initial_state()
    for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
        state = rules.apply(state, ChessMove(move))
    encoder = BoardEncoder(rules)

    assert encoder.encode_board(rules.board(state)) == encoder.encode(state)


def test_state_encoding_cache_reuses_immutable_result() -> None:
    rules = PythonChessRules()
    state = rules.apply(rules.initial_state(), ChessMove("e2e4"))
    encoder = BoardEncoder(rules, cache_size=1)

    assert encoder.encode(state) is encoder.encode(state)
    with pytest.raises(ValueError, match="cache size"):
        BoardEncoder(cache_size=0)
