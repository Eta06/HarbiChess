"""Prior-preserving classical value plus 224 zero-initialized PST residuals."""

import json
import math
from pathlib import Path

from frozen_reference.value import (
    FEATURE_SCALES,
    PRIOR,
    SCALE,
    ClassicalValue,
    features,
)
from frozen_reference.value import (
    model_dict as linear_model_dict,
)
from pst_features import FEATURE_COUNT, piece_square_features

SCHEMA = "classical-own-pst-residual-v1"
PST_L2 = 0.02
PST_SMOOTHNESS = 0.01


def model_dict(theta=None, pst_theta=None):
    if theta is None:
        theta = [0.0] * 18
    if pst_theta is None:
        pst_theta = [0.0] * FEATURE_COUNT
    return {
        "schema": SCHEMA,
        "linear": linear_model_dict(theta),
        "pst_feature_count": FEATURE_COUNT,
        "pst_tables": {
            "piece_types": ["pawn", "knight", "bishop", "rook", "queen"],
            "king_phase_tables": ["middlegame", "endgame"],
            "oriented_squares": "rank8xfile4-mirror-files-a-h",
        },
        "pst_l2": PST_L2,
        "pst_smoothness": PST_SMOOTHNESS,
        "pst_theta": list(pst_theta),
    }


class PSTValue:
    def __init__(self, theta=None, pst_theta=None):
        self.linear = ClassicalValue(theta=theta)
        self.pst_theta = tuple(
            float(x)
            for x in (pst_theta if pst_theta is not None else [0.0] * FEATURE_COUNT)
        )
        if len(self.pst_theta) != FEATURE_COUNT or not all(
            map(math.isfinite, self.pst_theta)
        ):
            raise ValueError("finite 224-parameter PST residual required")

    def nonterminal(self, board):
        base = self.linear.nonterminal(board)
        # Preserve exact zero-residual output instead of introducing tanh rounding.
        if not any(self.pst_theta):
            return base
        phi = piece_square_features(board)
        residual = sum(w * x for w, x in zip(self.pst_theta, phi, strict=True))
        raw = features(board)
        prior = sum(w * x for w, x in zip(PRIOR, raw, strict=True)) / SCALE
        linear_residual = sum(
            t * x / scale
            for t, x, scale in zip(self.linear.theta, raw, FEATURE_SCALES, strict=True)
        )
        return math.tanh(prior + linear_residual + residual)

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return (
                0.0
                if outcome.winner is None
                else (1.0 if outcome.winner == board.turn else -1.0)
            )
        return self.nonterminal(board)


def load_pst(path):
    data = json.loads(Path(path).read_text())
    if set(data) != set(model_dict()):
        raise ValueError("PST candidate fields differ")
    if data.get("schema") != SCHEMA or data.get("pst_feature_count") != FEATURE_COUNT:
        raise ValueError("PST candidate schema differs")
    if data.get("pst_tables") != model_dict()["pst_tables"]:
        raise ValueError("PST orientation/phase convention differs")
    if data.get("pst_l2") != PST_L2 or data.get("pst_smoothness") != PST_SMOOTHNESS:
        raise ValueError("PST regularization contract differs")
    linear = data["linear"]
    if linear != linear_model_dict(linear.get("theta")):
        raise ValueError("embedded linear prior/feature schema differs")
    values = data["pst_theta"]
    if len(values) != FEATURE_COUNT or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("PST parameter vector invalid")
    return PSTValue(linear["theta"], values)
