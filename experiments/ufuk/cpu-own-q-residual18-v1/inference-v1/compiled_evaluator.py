"""Immutable compiled inference; ordinary NumPy training and native state preserved."""
import math
import struct

import numpy as np
from _ownq_forward18 import forward


class CompiledEvaluator18:
    def __init__(self, model, *, classical_features, prior_weights, feature_scales,
                 prior_score_scale):
        shapes = {"w1": (18, 32), "b1": (32,), "w2": (32, 1), "b2": (1,)}
        if set(model.params) != set(shapes):
            raise ValueError("exact 18x32 parameter inventory required")
        for key, shape in shapes.items():
            p = np.asarray(model.params[key], dtype=np.float64)
            if p.shape != shape or not np.isfinite(p).all():
                raise ValueError("finite exact parameter shapes required")
        scales = np.asarray(feature_scales, dtype=np.float64)
        prior = np.asarray(prior_weights, dtype=np.float64)
        if scales.shape != (18,) or prior.shape != (18,) or not np.isfinite(scales).all() or not np.isfinite(prior).all() or (scales <= 0).any() or not math.isfinite(prior_score_scale) or prior_score_scale <= 0:
            raise ValueError("finite positive immutable prior constants required")
        values = [*(model.params["w1"] / scales[:, None]).ravel().tolist(),
                  *model.params["b1"].tolist(), *model.params["w2"][:, 0].tolist(),
                  float(model.params["b2"][0]), *prior.tolist(), float(prior_score_scale)]
        if len(values) != 660 or struct.calcsize("=660d") != 5280:
            raise ValueError("binary layout differs")
        self.payload = struct.pack("=660d", *values)
        self.features = classical_features

    def nonterminal(self, board):
        return forward(self.features(board), self.payload)

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0.0 if outcome.winner is None else 1.0 if outcome.winner == board.turn else -1.0
        return self.nonterminal(board)
