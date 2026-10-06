"""Zero-initialized coarse piece-square residual over the frozen classical prior."""

from __future__ import annotations

import math

PIECES = 5
ZONES = 16
PARAMS = PIECES * ZONES
ARCH = "coarse-piece-square-80-residual-v1"


def _check_theta(theta):
    if len(theta) != PARAMS or not all(math.isfinite(float(v)) for v in theta):
        raise ValueError("finite 80-parameter residual required")


def features_from_masks(masks, mover_white):
    """Signed type-by-4-by-4 occupancy; masks are black P..Q then white P..Q."""
    if len(masks) != 10 or type(mover_white) is not bool:
        raise ValueError("ten non-king bitboards and mover color required")
    x = [0.0] * PARAMS
    for color_index in range(2):
        sign = 1.0 if (color_index == 1) == mover_white else -1.0
        for piece_index in range(PIECES):
            bits = int(masks[color_index * PIECES + piece_index])
            if bits < 0 or bits >= 1 << 64:
                raise ValueError("piece mask is not uint64")
            while bits:
                low = bits & -bits
                square = low.bit_length() - 1
                bits ^= low
                oriented = square if color_index == 1 else square ^ 56
                rank, file = divmod(oriented, 8)
                zone = (rank // 2) * 4 + file // 2
                x[piece_index * ZONES + zone] += sign
    return tuple(x)


def features_from_board(board):
    import chess

    masks = [
        board.pieces_mask(piece, color)
        for color in (chess.BLACK, chess.WHITE)
        for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    ]
    return features_from_masks(masks, bool(board.turn)), tuple(masks)


def project_zero_mean(theta):
    """Center each type table; set the last bin as negative prefix sum."""
    if len(theta) != PARAMS:
        raise ValueError("80 parameters required")
    out = [float(v) for v in theta]
    for p in range(PIECES):
        start = p * ZONES
        mean = sum(out[start : start + ZONES]) / ZONES
        for j in range(ZONES - 1):
            out[start + j] -= mean
        out[start + ZONES - 1] = -sum(out[start : start + ZONES - 1])
    _check_theta(out)
    return out


def is_zero_mean(theta):
    if len(theta) != PARAMS:
        return False
    return all(
        sum(theta[p * ZONES : p * ZONES + ZONES - 1]) + theta[p * ZONES + ZONES - 1] == 0.0
        for p in range(PIECES)
    )


def residual_python(theta, features):
    _check_theta(theta)
    if len(features) != PARAMS or not all(math.isfinite(float(v)) for v in features):
        raise ValueError("80 features required")
    return sum(float(w) * float(x) for w, x in zip(theta, features, strict=True))


def combine_value(prior_value, prior_logit, residual):
    """Keep the old evaluator's exact output bits on zero residual."""
    return prior_value if residual == 0.0 else math.tanh(prior_logit + residual)


class Coarse80Evaluator:
    def __init__(self, prior, theta, classical_features=None, compiled=None):
        _check_theta(theta)
        self.prior = prior
        self.classical_features = classical_features or getattr(prior, "features", None)
        if self.classical_features is None:
            raise ValueError("pinned classical feature helper required")
        self.theta = tuple(float(v) for v in theta)
        self.compiled = compiled
        self.is_zero = all(v == 0.0 for v in self.theta)

    def nonterminal(self, board):
        if self.is_zero:
            return self.prior.nonterminal(board)
        raw = self.classical_features(board)
        prior_logit = (
            sum(w * x for w, x in zip(self.prior.weights, raw, strict=True)) / self.prior.scale
        )
        phi, masks = features_from_board(board)
        residual = (
            self.compiled(self.theta, masks, bool(board.turn))
            if self.compiled is not None
            else residual_python(self.theta, phi)
        )
        if residual == 0.0:
            return self.prior.nonterminal(board)
        result = math.tanh(prior_logit + residual)
        if not math.isfinite(result) or not -1.0 <= result <= 1.0:
            raise ValueError("invalid value")
        return result

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0.0 if outcome.winner is None else 1.0 if outcome.winner == board.turn else -1.0
        return self.nonterminal(board)
