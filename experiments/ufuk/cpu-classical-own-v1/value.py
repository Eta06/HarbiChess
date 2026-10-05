"""Transparent classical prior plus learnable scalar; no teacher or strength claim.

Positions are exact python-chess full-history boards. Search owns exact terminal
ordering; this standalone API also returns exact terminal mover W-L in [-1,1].
Pseudo-attack mobility is explicitly distinct from legal-move mobility.
"""

import math

import chess

SCHEMA = "classical-own-linear-value-v1"
NAMES = (
    "pawn",
    "knight",
    "bishop",
    "rook",
    "queen",
    "bishop_pair",
    "pawn_advance",
    "knight_center",
    "bishop_center",
    "rook_seventh",
    "queen_center",
    "king_center_end",
    "king_center_mid",
    "doubled_pawn",
    "isolated_pawn",
    "passed_pawn_advance",
    "attack_mobility",
    "king_ring_danger",
)
PRIOR = (
    100.0,
    320.0,
    330.0,
    500.0,
    900.0,
    30.0,
    5.0,
    14.0,
    8.0,
    18.0,
    2.0,
    18.0,
    -16.0,
    -12.0,
    -10.0,
    10.0,
    2.0,
    -8.0,
)
SCALE = 600.0
FEATURE_SCALES = (
    8.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    24.0,
    7.0,
    7.0,
    2.0,
    3.5,
    3.5,
    3.5,
    7.0,
    8.0,
    48.0,
    100.0,
    9.0,
)
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def features(board):
    """Signed side-to-move features, color-reflection symmetric, no board mutation."""
    phase = (
        min(
            24,
            sum(
                w * board.pieces_mask(p, c).bit_count()
                for p, w in PHASE_WEIGHTS.items()
                for c in (False, True)
            ),
        )
        / 24
    )
    x = [0.0] * len(NAMES)
    for color in (board.turn, not board.turn):
        sign = 1.0 if color == board.turn else -1.0
        own = board.occupied_co[color]
        pawns = board.pieces_mask(chess.PAWN, color)
        enemy_pawns = board.pieces_mask(chess.PAWN, not color)
        pawn_files = [(pawns & chess.BB_FILES[f]).bit_count() for f in range(8)]
        for p in range(chess.PAWN, chess.KING + 1):
            bits = board.pieces_mask(p, color)
            if p <= chess.QUEEN:
                x[p - 1] += sign * bits.bit_count()
            if p == chess.BISHOP and bits.bit_count() >= 2:
                x[5] += sign
            while bits:
                bit = bits & -bits
                bits ^= bit
                square = bit.bit_length() - 1
                oriented = square if color else square ^ 56
                rank, file = divmod(oriented, 8)
                center = 3.5 - (abs(file - 3.5) + abs(rank - 3.5)) / 2
                if p == chess.PAWN:
                    x[6] += sign * max(0, rank - 1)
                    isolated = not any(pawn_files[f] for f in (file - 1, file + 1) if 0 <= f < 8)
                    x[14] += sign * isolated
                    neighboring = chess.BB_FILES[file]
                    if file > 0:
                        neighboring |= chess.BB_FILES[file - 1]
                    if file < 7:
                        neighboring |= chess.BB_FILES[file + 1]
                    ahead = 0
                    for r in range(rank + 1, 8):
                        actual_r = r if color else 7 - r
                        ahead |= chess.BB_RANKS[actual_r]
                    if not (enemy_pawns & neighboring & ahead):
                        x[15] += sign * rank
                elif p == chess.KNIGHT:
                    x[7] += sign * center
                elif p == chess.BISHOP:
                    x[8] += sign * center
                elif p == chess.ROOK:
                    x[9] += sign * (rank == 6)
                elif p == chess.QUEEN:
                    x[10] += sign * center
                else:
                    x[11] += sign * center * (1 - phase)
                    x[12] += sign * center * phase
                if p not in (chess.PAWN, chess.KING):
                    x[16] += sign * (board.attacks_mask(square) & ~own).bit_count()
        x[13] += sign * sum(max(0, n - 1) for n in pawn_files)
        king = board.king(color)
        if king is not None:
            ring = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
            for square in chess.scan_forward(ring):
                if board.is_attacked_by(not color, square):
                    x[17] += sign * phase
    return tuple(x)


class ClassicalValue:
    def __init__(self, weights=PRIOR, theta=None):
        self.weights = tuple(float(v) for v in weights)
        if len(self.weights) != len(NAMES) or not all(map(math.isfinite, self.weights)):
            raise ValueError("finite exact feature-weight vector required")
        self.theta = tuple(0.0 for _ in NAMES) if theta is None else tuple(float(v) for v in theta)
        if len(self.theta) != len(NAMES) or not all(map(math.isfinite, self.theta)):
            raise ValueError("finite exact learned residual vector required")

    def nonterminal(self, board):
        phi = features(board)
        prior = sum(w * x for w, x in zip(self.weights, phi, strict=True)) / SCALE
        residual = sum(t * x / s for t, x, s in zip(self.theta, phi, FEATURE_SCALES, strict=True))
        return math.tanh(prior + residual)

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0.0 if outcome.winner is None else 1.0 if outcome.winner == board.turn else -1.0
        return self.nonterminal(board)


def model_dict(theta=None):
    return dict(
        schema=SCHEMA,
        feature_names=list(NAMES),
        feature_scales=list(FEATURE_SCALES),
        prior_cp=list(PRIOR),
        score_scale=SCALE,
        theta=list(theta) if theta is not None else [0.0] * len(NAMES),
    )


def load_classical(path):
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    expected = model_dict(data["theta"])
    if data != expected:
        raise ValueError("model schema/prior/feature convention differs")
    return ClassicalValue(theta=data["theta"])
