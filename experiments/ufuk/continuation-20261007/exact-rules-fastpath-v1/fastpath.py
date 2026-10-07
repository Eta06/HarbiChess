"""Dormant exact standard-chess repetition prototype; original search is untouched."""

import collections
import importlib.util
from pathlib import Path

import chess

ORIGINAL = Path(
    "/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py"
)


def load_original():
    import hashlib
    import sys

    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != (
        "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    ):
        raise ValueError("original-search-source-mismatch")
    spec = importlib.util.spec_from_file_location("exact_rules_original_de53", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RepetitionBoard(chess.Board):
    """Search-local copy, only legal pushes and pops back to its initial cursor.

    Same python-chess private transposition key and irreversible boundary. No
    persisted counter, FEN-only cache, variant support or mutable shared state.
    Direct board edits/rootward pop beyond the initial cursor are unsupported.
    """

    @classmethod
    def from_board(cls, original):
        if type(original) is not chess.Board and type(original) is not cls:
            raise ValueError("standard-chess-board-required")
        plain = chess.Board.copy(original, stack=True)
        board = cls(None, chess960=original.chess960)
        board.__dict__.update(plain.__dict__)
        history = chess.Board.copy(original, stack=True)
        counts = collections.Counter([history._transposition_key()])
        while history.move_stack:
            move = chess.Board.pop(history)
            if history.is_irreversible(move):
                break
            counts[history._transposition_key()] += 1
        board._repetition_counts = counts
        board._repetition_undo = []
        return board

    def push(self, move):
        previous = self._repetition_counts
        irreversible = self.is_irreversible(move)
        super().push(move)
        self._repetition_undo.append(previous if irreversible else None)
        if irreversible:
            self._repetition_counts = collections.Counter()
        self._repetition_counts[self._transposition_key()] += 1

    def pop(self):
        if not self._repetition_undo:
            raise ValueError("cannot-pop-before-search-local-cursor")
        previous = self._repetition_undo.pop()
        key = self._transposition_key()
        self._repetition_counts[key] -= 1
        if not self._repetition_counts[key]:
            del self._repetition_counts[key]
        move = super().pop()
        if previous is not None:
            self._repetition_counts = previous
        return move

    def copy(self, *, stack=True):
        if stack is not True:
            raise ValueError("prototype-requires-full-history-copy")
        # Re-seed from an ordinary full-history copy: never expose counters.
        return type(self).from_board(self)

    def is_repetition(self, count=3):
        return self._repetition_counts[self._transposition_key()] >= count

    def can_claim_threefold_repetition(self):
        if self.is_repetition(3):
            return True
        for move in self.generate_legal_moves():
            self.push(move)
            try:
                if self.is_repetition(3):
                    return True
            finally:
                self.pop()
        return False


original = load_original()


class BudgetSearch(original.BudgetSearch):
    def search(self, position):
        return super().search(RepetitionBoard.from_board(position))
