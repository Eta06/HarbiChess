"""Immutable constants and an independent oracle for prior-anchored TD(0).

This experimental objective is not enabled by existing trainers. The actor
uses an expected-score advantage, clipped importance, and recorded legal
support. WDL supervision is a soft target KL, not a fabricated game result.
Coefficients are explicit: no paper hyperparameter is silently adopted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

ONLINE_OBJECTIVE_SCHEMA = "prior-anchored-one-ply-v1"


@dataclass(frozen=True, slots=True)
class OnlineObjectiveConfig:
    policy_anchor_weight: float
    value_target_weight: float
    value_anchor_weight: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.policy_anchor_weight,
                self.value_target_weight,
                self.value_anchor_weight,
            )
        ):
            raise ValueError("online objective coefficients must be finite and nonnegative")


class OnlineLoss(NamedTuple):
    total: object
    actor: object
    policy_anchor: object
    value_target: object
    value_anchor: object


@dataclass(frozen=True, slots=True)
class OnlineObjectiveTargets:
    legal_masks: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    importance: np.ndarray
    target_wdl: np.ndarray
    base_policy: np.ndarray
    base_wdl: np.ndarray

    def __post_init__(self) -> None:
        arrays = {}
        for name in self.__dataclass_fields__:
            value = np.array(getattr(self, name), copy=True, order="C")
            if name == "legal_masks":
                if value.dtype != np.bool_:
                    raise ValueError("legal masks must have boolean dtype")
            elif name == "actions":
                if value.dtype.kind not in "iu":
                    raise ValueError("chosen action indices must have integer dtype")
            elif value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError("target constants must be finite floating point arrays")
            value.flags.writeable = False
            arrays[name] = value
        mask = arrays["legal_masks"]
        if mask.ndim != 2 or min(mask.shape) <= 0 or not mask.any(axis=1).all():
            raise ValueError("every nonempty policy row must have legal support")
        rows, actions = mask.shape
        for name in ("actions", "advantages", "importance"):
            if arrays[name].shape != (rows,):
                raise ValueError(f"invalid {name} shape")
        chosen = arrays["actions"]
        if (
            (chosen < 0).any()
            or (chosen >= actions).any()
            or not mask[np.arange(rows), chosen].all()
        ):
            raise ValueError("every chosen action must be in its legal support")
        if (np.abs(arrays["advantages"]) > 1).any() or (arrays["importance"] < 0).any():
            raise ValueError("advantages must be within [-1,1] and importance nonnegative")
        for name, shape in (
            ("base_policy", (rows, actions)),
            ("target_wdl", (rows, 3)),
            ("base_wdl", (rows, 3)),
        ):
            value = arrays[name]
            if (
                value.shape != shape
                or (value < 0).any()
                or (value > 1).any()
                or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=1e-6)
            ):
                raise ValueError(f"{name} must contain normalized probability rows")
        if np.any(arrays["base_policy"][~mask] != 0):
            raise ValueError("base policy must assign zero mass to illegal actions")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)


def reference_online_loss(
    policy_logits: np.ndarray,
    value_logits: np.ndarray,
    targets: OnlineObjectiveTargets,
    config: OnlineObjectiveConfig,
) -> OnlineLoss:
    """Float64 log-sum-exp oracle; zero-probability KL terms are exactly zero."""
    policy = np.asarray(policy_logits, dtype=np.float64)
    value = np.asarray(value_logits, dtype=np.float64)
    if policy.shape != targets.legal_masks.shape or value.shape != targets.target_wdl.shape:
        raise ValueError("online logit and target shapes differ")
    if not np.isfinite(policy).all() or not np.isfinite(value).all():
        raise ValueError("online logits must be finite")

    def log_softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    log_policy = log_softmax(np.where(targets.legal_masks, policy, -np.inf))
    log_value = log_softmax(value)

    def kl(probabilities, log_model):
        support = probabilities > 0
        safe = np.where(support, probabilities, 1)
        return (probabilities * (np.log(safe) - np.where(support, log_model, 0))).sum(axis=1)

    actor = -np.mean(
        targets.importance
        * targets.advantages
        * log_policy[np.arange(policy.shape[0]), targets.actions]
    )
    anchor = np.mean(kl(targets.base_policy, log_policy))
    target = np.mean(targets.importance * kl(targets.target_wdl, log_value))
    base_value = np.mean(kl(targets.base_wdl, log_value))
    total = (
        actor
        + config.policy_anchor_weight * anchor
        + config.value_target_weight * target
        + config.value_anchor_weight * base_value
    )
    return OnlineLoss(*(float(x) for x in (total, actor, anchor, target, base_value)))
