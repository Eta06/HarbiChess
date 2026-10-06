"""Frozen-weight C80 evaluator; original scalar prior remains authoritative."""

import ctypes
import math
from coarse80 import is_zero_mean
from compiled80 import Compiled80


class FastEvaluator80:
    def __init__(self, prior, theta, *, classical_features, prior_scale, binary):
        if len(theta) != 80 or not all(math.isfinite(float(v)) for v in theta):
            raise ValueError("80 finite parameters required")
        if not is_zero_mean(theta) or any(prior.theta):
            raise ValueError("zero-mean residual and frozen zero human prior required")
        if not math.isfinite(prior_scale) or prior_scale <= 0:
            raise ValueError("finite positive prior scale")
        self.prior = prior
        self.features = classical_features
        self.scale = float(prior_scale)
        self.theta = tuple(float(x) for x in theta)
        self.zero = all(x == 0.0 for x in self.theta)
        self.weights = (ctypes.c_double * 80)(*self.theta)
        self.kernel = Compiled80(binary)
        self.mask_type = ctypes.c_uint64 * 10

    def nonterminal(self, board):
        if self.zero:
            return self.prior.nonterminal(board)
        raw = self.features(board)
        prior = (
            sum(w * x for w, x in zip(self.prior.weights, raw, strict=True))
            / self.scale
        )
        black, white = board.occupied_co
        masks = self.mask_type(
            board.pawns & black,
            board.knights & black,
            board.bishops & black,
            board.rooks & black,
            board.queens & black,
            board.pawns & white,
            board.knights & white,
            board.bishops & white,
            board.rooks & white,
            board.queens & white,
        )
        residual = float(self.kernel.function(self.weights, masks, int(board.turn)))
        result = math.tanh(prior + residual)
        if not math.isfinite(result):
            raise ValueError("nonfinite coarse80 mover value")
        return result

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return (
                0.0
                if outcome.winner is None
                else 1.0
                if outcome.winner == board.turn
                else -1.0
            )
        return self.nonterminal(board)
