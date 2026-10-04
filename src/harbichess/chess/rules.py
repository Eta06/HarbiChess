"""Deterministic chess rules backed by the proven python-chess library."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import chess

from harbichess.core.state import (
    ChessMove,
    ChessState,
    GameOutcome,
    Side,
    TerminalResult,
)


@dataclass(frozen=True, slots=True)
class PositionView:
    """Portable facts about the current position."""

    fen: str
    side_to_move: Side
    in_check: bool
    halfmove_clock: int
    fullmove_number: int


class IllegalMoveError(ValueError):
    """Raised when a transition is not legal in the supplied state."""


class PythonChessRules:
    """Reconstruct and advance games without leaking python-chess objects."""

    def __init__(self, *, board_cache_size: int = 512) -> None:
        if board_cache_size <= 0:
            raise ValueError("board cache size must be positive")
        self.board_cache_size = board_cache_size
        self._thread_local = threading.local()

    def _cache(self) -> OrderedDict[ChessState, chess.Board]:
        cache = getattr(self._thread_local, "board_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._thread_local.board_cache = cache
        return cache

    def _remember(self, state: ChessState, board: chess.Board) -> None:
        cache = self._cache()
        cache[state] = board
        cache.move_to_end(state)
        while len(cache) > self.board_cache_size:
            cache.popitem(last=False)

    def _cached_board(self, state: ChessState) -> chess.Board:
        cache = self._cache()
        cached = cache.get(state)
        if cached is not None:
            cache.move_to_end(state)
            return cached

        # Cold misses must not recursively fill the LRU with every prefix.
        # Many independent long actor cursors otherwise evict one another's
        # final boards. Reuse the nearest cached ancestor, preserving its stack,
        # then validate/replay only the missing suffix and remember this state.
        ancestor_ply = len(state.moves)
        board = None
        while ancestor_ply > 0:
            ancestor_ply -= 1
            ancestor = ChessState(state.root_fen, state.moves[:ancestor_ply])
            cached_ancestor = cache.get(ancestor)
            if cached_ancestor is not None:
                cache.move_to_end(ancestor)
                board = cached_ancestor.copy(stack=True)
                break
        if board is None:
            board = chess.Board(state.root_fen)
            ancestor_ply = 0
        for ply in range(ancestor_ply, len(state.moves)):
            encoded_move = state.moves[ply]
            try:
                move = chess.Move.from_uci(encoded_move.uci)
            except ValueError as error:
                raise IllegalMoveError(f"invalid move at ply {ply}: {encoded_move.uci}") from error
            if move not in board.legal_moves:
                raise IllegalMoveError(f"illegal move at ply {ply}: {encoded_move.uci}")
            board.push(move)
        self._remember(state, board)
        return board

    def initial_state(self, fen: str = chess.STARTING_FEN) -> ChessState:
        board = chess.Board(fen)
        state = ChessState(root_fen=board.fen())
        self._remember(state, board)
        return state

    def board(self, state: ChessState) -> chess.Board:
        """Return an isolated board while retaining an incremental internal cache."""

        return self._cached_board(state).copy(stack=True)

    def inspect(self, state: ChessState) -> chess.Board:
        """Borrow the cached board for read-only hot-path inspection.

        The returned object must never be mutated. State transitions must continue
        to use :meth:`apply`, which copies before pushing and keeps cache entries
        isolated from one another.
        """

        return self._cached_board(state)

    def view(self, state: ChessState) -> PositionView:
        board = self._cached_board(state)
        return PositionView(
            fen=board.fen(),
            side_to_move=Side.WHITE if board.turn is chess.WHITE else Side.BLACK,
            in_check=board.is_check(),
            halfmove_clock=board.halfmove_clock,
            fullmove_number=board.fullmove_number,
        )

    def legal_moves(self, state: ChessState) -> tuple[ChessMove, ...]:
        board = self._cached_board(state)
        return tuple(ChessMove(move.uci()) for move in sorted(board.legal_moves, key=str))

    def apply(self, state: ChessState, move: ChessMove) -> ChessState:
        result = ChessState(root_fen=state.root_fen, moves=(*state.moves, move))
        cache = self._cache()
        if result in cache:
            cache.move_to_end(result)
            return result
        board = self._cached_board(state).copy(stack=True)
        parsed = chess.Move.from_uci(move.uci)
        if parsed not in board.legal_moves:
            raise IllegalMoveError(f"illegal move: {move.uci}")
        board.push(parsed)
        self._remember(result, board)
        return result

    def outcome(self, state: ChessState, *, claim_draw: bool = False) -> GameOutcome | None:
        board = self._cached_board(state)
        outcome = board.outcome(claim_draw=claim_draw)
        if outcome is None:
            return None
        result = TerminalResult(outcome.result())
        return GameOutcome(result=result, termination=outcome.termination.name.lower())

    def claimable_threefold_moves(
        self,
        state: ChessState,
        moves: tuple[ChessMove, ...],
    ) -> frozenset[ChessMove]:
        """Return supplied moves after which a threefold claim is available."""

        if state.ply < 6 or not moves:
            return frozenset()
        board = self._cached_board(state).copy(stack=True)
        repeating = []
        for move in moves:
            parsed = chess.Move.from_uci(move.uci)
            if parsed not in board.legal_moves:
                raise IllegalMoveError(f"illegal move: {move.uci}")
            board.push(parsed)
            if board.is_repetition(3) or board.can_claim_threefold_repetition():
                repeating.append(move)
            board.pop()
        return frozenset(repeating)
