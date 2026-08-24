"""History-aware, side-to-move canonical board encoding."""

from __future__ import annotations

import threading
from collections import OrderedDict

import chess

from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition
from harbichess.core.state import ChessState

HISTORY_STEPS = 8
PIECE_PLANES_PER_STEP = 12
METADATA_PLANES = 8
ENCODER_CHANNELS = HISTORY_STEPS * PIECE_PLANES_PER_STEP + METADATA_PLANES
ENCODER_SCHEMA_VERSION = 1


def _canonical_square(square: chess.Square, perspective: chess.Color) -> chess.Square:
    return square if perspective is chess.WHITE else chess.square_mirror(square)


class BoardEncoder:
    """Encode current and recent positions into an NHWC-compatible tensor."""

    def __init__(
        self,
        rules: PythonChessRules | None = None,
        *,
        cache_size: int = 64,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("encoding cache size must be positive")
        self._rules = rules or PythonChessRules()
        self._cache_size = cache_size
        self._thread_local = threading.local()

    def _cache(self) -> OrderedDict[ChessState, EncodedPosition]:
        cache = getattr(self._thread_local, "encoding_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._thread_local.encoding_cache = cache
        return cache

    def encode(self, state: ChessState) -> EncodedPosition:
        return self.encode_state(state, self._rules.inspect(state))

    def encode_state(self, state: ChessState, board: chess.Board) -> EncodedPosition:
        """Encode and retain an immutable state using an already inspected board."""

        cache = self._cache()
        encoded = cache.get(state)
        if encoded is not None:
            cache.move_to_end(state)
            return encoded
        encoded = self.encode_board(board)
        cache[state] = encoded
        while len(cache) > self._cache_size:
            cache.popitem(last=False)
        return encoded

    def encode_board(self, board: chess.Board) -> EncodedPosition:
        """Encode a read-only board without reconstructing its move history."""

        perspective = board.turn
        values = [0.0] * (8 * 8 * ENCODER_CHANNELS)

        historical = board.copy(stack=HISTORY_STEPS - 1)
        for step in range(HISTORY_STEPS):
            if step > 0:
                if not historical.move_stack:
                    break
                historical.pop()
            self._write_piece_planes(values, historical, perspective, step)

        metadata_start = HISTORY_STEPS * PIECE_PLANES_PER_STEP
        self._fill_plane(values, metadata_start, 1.0 if perspective is chess.WHITE else 0.0)
        self._fill_plane(
            values,
            metadata_start + 1,
            float(board.has_kingside_castling_rights(perspective)),
        )
        self._fill_plane(
            values,
            metadata_start + 2,
            float(board.has_queenside_castling_rights(perspective)),
        )
        opponent = not perspective
        self._fill_plane(
            values,
            metadata_start + 3,
            float(board.has_kingside_castling_rights(opponent)),
        )
        self._fill_plane(
            values,
            metadata_start + 4,
            float(board.has_queenside_castling_rights(opponent)),
        )
        if board.ep_square is not None:
            square = _canonical_square(board.ep_square, perspective)
            self._set(values, square, metadata_start + 5, 1.0)
        self._fill_plane(values, metadata_start + 6, min(board.halfmove_clock, 100) / 100)
        repetition = 1.0 if board.is_repetition(3) else 0.5 if board.is_repetition(2) else 0.0
        self._fill_plane(values, metadata_start + 7, repetition)

        return EncodedPosition(
            values=tuple(values),
            shape=(8, 8, ENCODER_CHANNELS),
            schema_version=ENCODER_SCHEMA_VERSION,
        )

    @staticmethod
    def _write_piece_planes(
        values: list[float],
        board: chess.Board,
        perspective: chess.Color,
        step: int,
    ) -> None:
        offset = step * PIECE_PLANES_PER_STEP
        for square, piece in board.piece_map().items():
            canonical = _canonical_square(square, perspective)
            side_offset = 0 if piece.color is perspective else 6
            channel = offset + side_offset + piece.piece_type - 1
            BoardEncoder._set(values, canonical, channel, 1.0)

    @staticmethod
    def _set(values: list[float], square: chess.Square, channel: int, value: float) -> None:
        values[square * ENCODER_CHANNELS + channel] = value

    @staticmethod
    def _fill_plane(values: list[float], channel: int, value: float) -> None:
        if value == 0.0:
            return
        for square in chess.SQUARES:
            BoardEncoder._set(values, square, channel, value)
