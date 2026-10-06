"""Frozen Python prior authoritative; C computes only learned residual."""

import math

import torch
from native import validate_weights


class Evaluator:
    def __init__(self, state, *, prior, compiled):
        validate_weights(state)
        self.prior, self.compiled = prior, compiled
        self.zero = not bool(
            torch.count_nonzero(state["head.weight"]) or torch.count_nonzero(state["head.bias"])
        )
        self.payload = (
            torch.cat(
                [
                    state["embedding.weight"].reshape(-1),
                    state["head.weight"].reshape(-1),
                    state["head.bias"],
                ]
            )
            .contiguous()
            .numpy()
            .tobytes()
        )
        if len(self.payload) != 196625 * 8:
            raise ValueError("exact compiled model geometry")

    def nonterminal(self, board):
        if self.zero:
            # Literal Python builtin sum / original tanh path, never reconstructed in C.
            return self.prior.nonterminal(board)

        king = board.king(board.turn)
        bitboards = [
            board.pieces_mask(piece, color) for color in [True, False] for piece in range(1, 7)
        ]
        residual = self.compiled.forward(bitboards, king, board.turn, self.payload)
        if residual == 0.0:
            return self.prior.nonterminal(board)
        # The authoritative original Python dot product is preferable to atanh
        # roundtrip: inject original prior-logit function for trained inference.
        if not hasattr(self.prior, "logit"):
            raise ValueError("trained evaluator requires pinned original prior.logit")
        return math.tanh(self.prior.logit(board) + residual)

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return (
                0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
            )
        return self.nonterminal(board)


class AuthoritativePrior:
    """Pinned original Python feature/sum/tanh implementation, zero theta only."""

    def __init__(self, module, original):
        if original.weights != tuple(module.PRIOR) or any(original.theta):
            raise ValueError("immutable humanprior18 only")
        self.module, self.original = module, original

    def nonterminal(self, board):
        return self.original.nonterminal(board)

    def logit(self, board):
        phi = self.module.features(board)
        return sum(w * x for w, x in zip(self.module.PRIOR, phi, strict=True)) / self.module.SCALE
