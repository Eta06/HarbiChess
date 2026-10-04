"""Differentiable MLX implementation; actual CPU/device parity is testable."""

import mlx.core as mx

from harbichess.training.online_objective import (
    OnlineLoss,
    OnlineObjectiveConfig,
    OnlineObjectiveTargets,
)


def online_loss(
    policy_logits: mx.array,
    value_logits: mx.array,
    targets: OnlineObjectiveTargets,
    config: OnlineObjectiveConfig,
) -> OnlineLoss:
    if (
        tuple(policy_logits.shape) != targets.legal_masks.shape
        or tuple(value_logits.shape) != targets.target_wdl.shape
        or policy_logits.dtype != value_logits.dtype
        or policy_logits.dtype != mx.float32
    ):
        raise ValueError("MLX online logits require matching float32 shapes and dtype")
    if not bool(mx.all(mx.isfinite(policy_logits))) or not bool(mx.all(mx.isfinite(value_logits))):
        raise ValueError("online logits must be finite")

    def constant(array, dtype=mx.float32):
        return mx.stop_gradient(mx.array(array, dtype=dtype))

    mask = constant(targets.legal_masks, mx.bool_)
    chosen = constant(targets.actions, mx.int32)
    importance = constant(targets.importance)
    masked = mx.where(mask, policy_logits, -mx.inf)
    log_policy = masked - mx.logsumexp(masked, axis=1, keepdims=True)
    log_value = value_logits - mx.logsumexp(value_logits, axis=1, keepdims=True)

    def kl(probabilities, log_model):
        support = probabilities > 0
        safe = mx.where(support, probabilities, 1.0)
        return (probabilities * (mx.log(safe) - mx.where(support, log_model, 0.0))).sum(axis=1)

    selected = mx.take_along_axis(log_policy, chosen[:, None], axis=1)[:, 0]
    actor = -(importance * constant(targets.advantages) * selected).mean()
    anchor = kl(constant(targets.base_policy), log_policy).mean()
    target = (importance * kl(constant(targets.target_wdl), log_value)).mean()
    base_value = kl(constant(targets.base_wdl), log_value).mean()
    total = (
        actor
        + config.policy_anchor_weight * anchor
        + config.value_target_weight * target
        + config.value_anchor_weight * base_value
    )
    return OnlineLoss(total, actor, anchor, target, base_value)
