"""Same full-root/minimax controller for zero/trained effort, charged512 ceiling."""

import math

from model import features, load_model, model_dict, probability

BASE_SHA = "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"


def search_type(base, exhausted):
    class Selective(base):
        def __init__(self, evaluator, *, weights=None, **kwargs):
            super().__init__(evaluator, **kwargs)
            if self.qdepth != 2:
                raise ValueError("frozen nominalq2")
            self.weights = load_model(model_dict([0.0] * 16 if weights is None else weights))
            self.extension_receipts = []

        def extra(self, board, alpha, beta, remaining, ply, start):
            if self.nodes - start >= 64:
                raise exhausted()
            self.consume()
            terminal = self.terminal(board, ply)
            if terminal is not None:
                return terminal
            checked = board.is_check()
            if remaining == 0 and checked:
                raise exhausted()
            static = self.evaluate(board)
            if remaining == 0:
                return static
            best = -math.inf if checked else static
            if not checked:
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            moves = list(board.legal_moves)
            if not checked:
                moves = [m for m in moves if board.is_capture(m) or m.promotion]
            for move in self.ordered(board, moves):
                board.push(move)
                try:
                    v = -self.extra(board, -beta, -alpha, remaining - 1, ply + 1, start)
                finally:
                    board.pop()
                best = max(best, v)
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return best

        def quiesce(self, board, alpha, beta, remaining, ply):
            if remaining != 0:
                return super().quiesce(board, alpha, beta, remaining, ply)
            self.consume()
            terminal = self.terminal(board, ply)
            if terminal is not None:
                return terminal
            checked = board.is_check()
            # No static stand-pat in check in either new arm. If bounded
            # evasion search fails, discard the incomplete ROOT pass upstream.
            static = None if checked else self.evaluate(board)
            chosen = checked or probability(self.weights, features(board, static)) > 0.5
            if not chosen:
                return static
            start = self.nodes
            try:
                value = self.extra(board, alpha, beta, 2, ply, start)
            except exhausted:
                self.extension_receipts.append(
                    dict(nodes=self.nodes - start, status="censored", checked=checked)
                )
                raise
            self.extension_receipts.append(
                dict(nodes=self.nodes - start, status="completed", checked=checked)
            )
            return value

    return Selective
