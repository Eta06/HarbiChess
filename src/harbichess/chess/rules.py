"""Deterministic chess rules backed by the proven python-chess library."""

from __future__ import annotations

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

    def initial_state(self, fen: str = chess.STARTING_FEN) -> ChessState:
        board = chess.Board(fen)
        return ChessState(root_fen=board.fen())

    def board(self, state: ChessState) -> chess.Board:
        board = chess.Board(state.root_fen)
        for ply, encoded_move in enumerate(state.moves):
            try:
                move = chess.Move.from_uci(encoded_move.uci)
            except ValueError as error:
                raise IllegalMoveError(f"invalid move at ply {ply}: {encoded_move.uci}") from error
            if move not in board.legal_moves:
                raise IllegalMoveError(f"illegal move at ply {ply}: {encoded_move.uci}")
            board.push(move)
        return board

    def view(self, state: ChessState) -> PositionView:
        board = self.board(state)
        return PositionView(
            fen=board.fen(),
            side_to_move=Side.WHITE if board.turn is chess.WHITE else Side.BLACK,
            in_check=board.is_check(),
            halfmove_clock=board.halfmove_clock,
            fullmove_number=board.fullmove_number,
        )

    def legal_moves(self, state: ChessState) -> tuple[ChessMove, ...]:
        board = self.board(state)
        return tuple(ChessMove(move.uci()) for move in sorted(board.legal_moves, key=str))

    def apply(self, state: ChessState, move: ChessMove) -> ChessState:
        board = self.board(state)
        parsed = chess.Move.from_uci(move.uci)
        if parsed not in board.legal_moves:
            raise IllegalMoveError(f"illegal move: {move.uci}")
        return ChessState(root_fen=state.root_fen, moves=(*state.moves, move))

    def outcome(self, state: ChessState, *, claim_draw: bool = False) -> GameOutcome | None:
        board = self.board(state)
        outcome = board.outcome(claim_draw=claim_draw)
        if outcome is None:
            return None
        result = TerminalResult(outcome.result())
        return GameOutcome(result=result, termination=outcome.termination.name.lower())

