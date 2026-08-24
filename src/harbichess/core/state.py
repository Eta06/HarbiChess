"""Framework-independent chess state and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    """A chess side, encoded without depending on a rules library."""

    WHITE = "white"
    BLACK = "black"


class TerminalResult(StrEnum):
    """Terminal game result in portable PGN-compatible form."""

    WHITE_WIN = "1-0"
    BLACK_WIN = "0-1"
    DRAW = "1/2-1/2"


@dataclass(frozen=True, slots=True)
class ChessMove:
    """A canonical move encoded as lower-case UCI text."""

    uci: str

    def __post_init__(self) -> None:
        normalized = self.uci.strip().lower()
        if len(normalized) not in (4, 5):
            raise ValueError(f"invalid UCI move length: {self.uci!r}")
        object.__setattr__(self, "uci", normalized)


@dataclass(frozen=True, slots=True)
class ChessState:
    """A lossless game state represented by a root FEN and its move history.

    A current-position FEN alone cannot reconstruct repetition history. Keeping
    the root plus the full legal move sequence preserves every state transition
    required by the rules engine and allows deterministic replay.
    """

    root_fen: str
    moves: tuple[ChessMove, ...] = ()

    @property
    def ply(self) -> int:
        return len(self.moves)


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """A terminal result and its rules-engine termination label."""

    result: TerminalResult
    termination: str

    def value_for(self, side: Side) -> int:
        """Return the terminal value from ``side``'s perspective."""

        if self.result is TerminalResult.DRAW:
            return 0
        winner = Side.WHITE if self.result is TerminalResult.WHITE_WIN else Side.BLACK
        return 1 if winner is side else -1

