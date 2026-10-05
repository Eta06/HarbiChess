"""Prevalidated 18-feature residual path for the fast-prior latency contingency."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from residual_critic18 import HIDDEN, INPUTS, ResidualCritic18


class ResidualEvaluator18:
    """Add an 18→32→1 residual to the frozen prior's own logit."""

    def __init__(
        self,
        model: ResidualCritic18,
        *,
        classical_features,
        prior_weights,
        feature_scales,
        prior_score_scale,
    ):
        if set(model.params) != {"w1", "b1", "w2", "b2"}:
            raise ValueError("18→32 residual parameter inventory differs")
        expected_shapes = {
            "w1": (INPUTS, HIDDEN),
            "b1": (HIDDEN,),
            "w2": (HIDDEN, 1),
            "b2": (1,),
        }
        for key, shape in expected_shapes.items():
            value = np.asarray(model.params[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid prevalidated residual tensor: {key}")
        self.classical_features = classical_features
        self.weights = tuple(float(value) for value in prior_weights)
        self.scales = np.asarray(feature_scales, dtype=np.float64)
        self.prior_score_scale = float(prior_score_scale)
        if (
            len(self.weights) != INPUTS
            or self.scales.shape != (INPUTS,)
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
            or not math.isfinite(self.prior_score_scale)
            or self.prior_score_scale <= 0
        ):
            raise ValueError("frozen 18-feature prior constants are invalid")
        self.w1 = model.params["w1"]
        self.b1 = model.params["b1"]
        self.w2 = model.params["w2"][:, 0]
        self.b2 = float(model.params["b2"][0])

    def nonterminal(self, board):
        raw = tuple(float(value) for value in self.classical_features(board))
        if len(raw) != INPUTS:
            raise ValueError("classical helper returned the wrong feature count")
        features = np.fromiter(
            (value / scale for value, scale in zip(raw, self.scales, strict=True)),
            dtype=np.float64,
            count=INPUTS,
        )
        prior_logit = sum(
            weight * value for weight, value in zip(self.weights, raw, strict=True)
        ) / self.prior_score_scale
        hidden = np.tanh(features @ self.w1 + self.b1)
        residual = float(hidden @ self.w2 + self.b2)
        logit = prior_logit if residual == 0.0 else prior_logit + residual
        output = math.tanh(float(logit))
        if not math.isfinite(output) or not -1.0 <= output <= 1.0:
            raise ValueError("18→32 residual evaluator produced an invalid value")
        return output

    def __call__(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0.0 if outcome.winner is None else 1.0 if outcome.winner == board.turn else -1.0
        return self.nonterminal(board)


def labels_to_groups18(payload, *, replay_position, classical_features, prior_weights,
                       feature_scales, prior_score_scale):
    """Convert exactly 1,024 frozen own-search labels to grouped 18-feature rows."""
    roots = payload.get("roots")
    if payload.get("schema") != "own-search-deeper-value-labels-v1" or not isinstance(roots, list):
        raise ValueError("registered own-search labels required")
    if len(roots) != 1024:
        raise ValueError("exactly 1,024 selected own-search roots required")
    groups = {}
    seen = set()
    for row in roots:
        if set(row) != {
            "ordinal", "row_id", "trajectory_id", "history_sha256", "history",
            "root_mover", "root_fen4", "raw_q_mover", "target_mover",
            "exact_mate_score_range", "zero_value_is_not_a_draw_certificate",
            "nodes", "evaluations", "completed_depth", "root_actions",
        }:
            raise ValueError("label row schema differs")
        identity = row["row_id"]
        if row["ordinal"] != len(seen) or not isinstance(identity, str) or identity in seen:
            raise ValueError("label identity/order differs")
        seen.add(identity)
        history = row["history"]
        root_fen, prefix = history.get("root_fen"), history.get("prefix_uci")
        history_digest = hashlib.sha256((root_fen + "\n" + " ".join(prefix)).encode()).hexdigest()
        if history_digest != row["history_sha256"]:
            raise ValueError("full-history digest differs")
        board = replay_position(root_fen, prefix)
        if (
            " ".join(board.fen().split()[:4]) != row["root_fen4"]
            or ("white" if board.turn else "black") != row["root_mover"]
        ):
            raise ValueError("replayed board or mover differs")
        raw_q = float(row["raw_q_mover"])
        target = max(-1.0, min(1.0, raw_q))
        if (
            not math.isfinite(raw_q)
            or abs(raw_q) > 2.0
            or target != row["target_mover"]
            or row["exact_mate_score_range"] != (abs(raw_q) > 1.0)
            or row["zero_value_is_not_a_draw_certificate"] != (raw_q == 0.0)
            or not 1 <= row["nodes"] <= 8192
            or not 0 <= row["evaluations"] <= row["nodes"]
            or not 0 <= row["completed_depth"] <= 8
        ):
            raise ValueError("own-search target bounds or semantics differ")
        raw = tuple(float(value) for value in classical_features(board))
        if len(raw) != INPUTS or not all(math.isfinite(value) for value in raw):
            raise ValueError("finite 18-feature input required")
        if len(prior_weights) != INPUTS or len(feature_scales) != INPUTS:
            raise ValueError("frozen 18-feature prior constants differ")
        x = [value / scale for value, scale in zip(raw, feature_scales, strict=True)]
        prior_logit = sum(
            weight * value for weight, value in zip(prior_weights, raw, strict=True)
        ) / prior_score_scale
        groups.setdefault(row["trajectory_id"], []).append(
            {"x": x, "prior_logit": float(prior_logit), "target": target}
        )
    if sum(map(len, groups.values())) != 1024:
        raise ValueError("label row coverage differs")
    return groups
